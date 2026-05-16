from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEG_SCRIPT = PROJECT_ROOT / "seg" / "seg.py"

DEFAULT_CROP_ROOT = PROJECT_ROOT / "first_data" / "enhence" / "crop"
DEFAULT_POSITIVE_ROOT = PROJECT_ROOT / "first_data" / "enhence" / "positive"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "first_data" / "enhence" / "concatenate_yolo"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = {
    0: "collision",
    1: "dirt",
    2: "plain_particle_dent",
    3: "plain_particle_particle",
    4: "scratch_damage",
    5: "scratch_mark",
}
CLASS_IDS = {name: idx for idx, name in CLASS_NAMES.items()}


@dataclass(frozen=True)
class CropAsset:
    path: Path
    class_id: int
    class_name: str


@dataclass
class Placement:
    class_id: int
    class_name: str
    target_path: Path
    x1: int
    y1: int
    x2: int
    y2: int
    rotation: float
    scale: float
    black_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose transparent target crops onto target-free backgrounds and "
            "write a YOLO detection dataset."
        )
    )
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--positive-root", type=Path, default=DEFAULT_POSITIVE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num", type=int, default=100, help="Debug default: 100 images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-targets", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--required-black-ratio", type=float, default=0.95)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.05)
    parser.add_argument("--max-place-attempts", type=int, default=400)
    parser.add_argument("--max-image-retries", type=int, default=30)
    parser.add_argument("--alpha-threshold", type=int, default=8)
    parser.add_argument(
        "--mask-erode",
        type=int,
        default=3,
        help="Erode clear black mask by this many pixels before placement.",
    )
    parser.add_argument(
        "--visualize-limit",
        type=int,
        default=100,
        help="Save visual debug previews for the first N generated images.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output folder before generating.",
    )
    return parser.parse_args()


def load_seg_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("seg_for_concatenate", SEG_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load segmentation script: {SEG_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seg_args() -> SimpleNamespace:
    return SimpleNamespace(
        blur_kernel=5,
        uncertainty_ratio=0.08,
        min_margin=8,
        max_margin=32,
        transition_radius=2,
        min_area=0,
        no_defocus=False,
        defocus_window=65,
        defocus_min_contrast=8.0,
        defocus_percentile=40.0,
        defocus_max_score=2.8,
        defocus_close=51,
        defocus_open=31,
        defocus_min_area_ratio=0.005,
        defocus_row_fill_ratio=0.18,
        defocus_min_row_run=0.08,
    )


def iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def load_crop_assets(crop_root: Path) -> list[CropAsset]:
    assets: list[CropAsset] = []
    for class_name, class_id in sorted(CLASS_IDS.items(), key=lambda item: item[1]):
        class_dir = crop_root / class_name
        for path in sorted(class_dir.glob("*.png")):
            assets.append(CropAsset(path=path, class_id=class_id, class_name=class_name))
    return assets


def clear_black_mask(image: Image.Image, seg_module: ModuleType, erode: int) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    result = seg_module.segment_adaptive(gray, seg_args())
    mask = result.black_mask.astype(np.uint8)
    if erode > 0:
        size = erode * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask.astype(bool)


def trim_alpha(image: Image.Image, alpha_threshold: int) -> Image.Image | None:
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > alpha_threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return rgba.crop(bbox)


def transform_crop(
    crop_path: Path,
    rng: random.Random,
    scale_min: float,
    scale_max: float,
    alpha_threshold: int,
) -> tuple[Image.Image, float, float] | None:
    with Image.open(crop_path) as raw:
        crop = trim_alpha(raw, alpha_threshold)
    if crop is None:
        return None

    scale = rng.uniform(scale_min, scale_max)
    new_size = (
        max(1, int(round(crop.width * scale))),
        max(1, int(round(crop.height * scale))),
    )
    crop = crop.resize(new_size, Image.Resampling.LANCZOS)

    rotation = rng.uniform(0.0, 360.0)
    crop = crop.rotate(
        rotation,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )
    crop = trim_alpha(crop, alpha_threshold)
    if crop is None:
        return None
    return crop, rotation, scale


def alpha_mask(image: Image.Image, alpha_threshold: int) -> np.ndarray:
    return np.asarray(image.getchannel("A")) > alpha_threshold


def find_position(
    target_mask: np.ndarray,
    clear_mask: np.ndarray,
    occupied_mask: np.ndarray,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[int, int, float] | None:
    bg_h, bg_w = clear_mask.shape
    target_h, target_w = target_mask.shape
    if target_w >= bg_w or target_h >= bg_h:
        return None

    candidate_pixels = np.argwhere(clear_mask)
    if len(candidate_pixels) == 0:
        return None

    target_area = int(target_mask.sum())
    if target_area == 0:
        return None

    for _ in range(args.max_place_attempts):
        center_y, center_x = candidate_pixels[rng.randrange(len(candidate_pixels))]
        anchor_x = rng.randrange(target_w)
        anchor_y = rng.randrange(target_h)
        left = int(center_x) - anchor_x
        top = int(center_y) - anchor_y
        if left < 0 or top < 0 or left + target_w > bg_w or top + target_h > bg_h:
            continue

        clear_region = clear_mask[top : top + target_h, left : left + target_w]
        black_ratio = float((clear_region & target_mask).sum()) / target_area
        if black_ratio < args.required_black_ratio:
            continue

        occupied_region = occupied_mask[top : top + target_h, left : left + target_w]
        overlap_ratio = float((occupied_region & target_mask).sum()) / target_area
        if overlap_ratio > args.max_overlap_ratio:
            continue

        return left, top, black_ratio

    return None


def paste_target(
    canvas: Image.Image,
    target: Image.Image,
    target_mask: np.ndarray,
    left: int,
    top: int,
    occupied_mask: np.ndarray,
) -> tuple[int, int, int, int]:
    canvas.alpha_composite(target, (left, top))
    h, w = target_mask.shape
    occupied_mask[top : top + h, left : left + w] |= target_mask
    ys, xs = np.where(target_mask)
    x1 = left + int(xs.min())
    y1 = top + int(ys.min())
    x2 = left + int(xs.max()) + 1
    y2 = top + int(ys.max()) + 1
    return x1, y1, x2, y2


def yolo_line(placement: Placement, width: int, height: int) -> str:
    x_center = ((placement.x1 + placement.x2) / 2.0) / width
    y_center = ((placement.y1 + placement.y2) / 2.0) / height
    box_w = (placement.x2 - placement.x1) / width
    box_h = (placement.y2 - placement.y1) / height
    return (
        f"{placement.class_id} "
        f"{x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
    )


def draw_preview(
    image: Image.Image,
    clear_mask: np.ndarray,
    placements: list[Placement],
) -> Image.Image:
    boxed = image.convert("RGB")
    draw = ImageDraw.Draw(boxed)
    colors = {
        0: (255, 80, 80),
        1: (255, 180, 40),
        2: (60, 220, 120),
        3: (40, 180, 255),
        4: (180, 120, 255),
        5: (255, 80, 200),
    }

    for placement in placements:
        color = colors[placement.class_id]
        draw.rectangle(
            [placement.x1, placement.y1, placement.x2, placement.y2],
            outline=color,
            width=3,
        )
        text = f"{placement.class_id}:{placement.class_name}"
        text_bbox = draw.textbbox((placement.x1, placement.y1), text)
        tx1, ty1, tx2, ty2 = text_bbox
        label_y1 = max(0, placement.y1 - (ty2 - ty1) - 4)
        draw.rectangle(
            [placement.x1, label_y1, placement.x1 + (tx2 - tx1) + 6, label_y1 + (ty2 - ty1) + 4],
            fill=color,
        )
        draw.text((placement.x1 + 3, label_y1 + 2), text, fill=(0, 0, 0))

    mask_rgb = image.convert("RGB")
    mask_overlay = Image.new("RGB", mask_rgb.size, (0, 255, 80))
    mask_alpha = Image.fromarray((clear_mask.astype(np.uint8) * 105), mode="L")
    mask_rgb.paste(mask_overlay, (0, 0), mask_alpha)

    preview = Image.new("RGB", (boxed.width * 2, boxed.height), (0, 0, 0))
    preview.paste(boxed, (0, 0))
    preview.paste(mask_rgb, (boxed.width, 0))
    return preview


def write_data_yaml(output_root: Path) -> None:
    lines = [
        f"path: {output_root}",
        "",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in CLASS_NAMES.items())
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_for_index(index: int, total: int, val_ratio: float) -> str:
    val_count = int(round(total * val_ratio))
    if val_count <= 0:
        return "train"
    return "val" if index > total - val_count else "train"


def prepare_output(output_root: Path, clean: bool) -> None:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "visualize" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "placement_masks" / split).mkdir(parents=True, exist_ok=True)


def save_placement_mask(path: Path, clear_mask: np.ndarray) -> None:
    mask = Image.fromarray((clear_mask.astype(np.uint8) * 255), mode="L")
    mask.save(path)


def compose_one(
    index: int,
    background_paths: list[Path],
    crop_assets: list[CropAsset],
    seg_module: ModuleType,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[Image.Image, np.ndarray, list[Placement], Path]:
    for _ in range(args.max_image_retries):
        background_path = rng.choice(background_paths)
        with Image.open(background_path) as bg:
            background = bg.convert("RGB")

        clear_mask = clear_black_mask(background, seg_module, args.mask_erode)
        if not clear_mask.any():
            continue

        canvas = background.convert("RGBA")
        occupied_mask = np.zeros(clear_mask.shape, dtype=bool)
        placements: list[Placement] = []
        target_count = rng.randint(args.min_targets, args.max_targets)

        for _target_index in range(target_count):
            asset = rng.choice(crop_assets)
            transformed = transform_crop(
                asset.path,
                rng,
                args.scale_min,
                args.scale_max,
                args.alpha_threshold,
            )
            if transformed is None:
                continue

            target, rotation, scale = transformed
            target_mask = alpha_mask(target, args.alpha_threshold)
            position = find_position(target_mask, clear_mask, occupied_mask, rng, args)
            if position is None:
                continue

            left, top, black_ratio = position
            x1, y1, x2, y2 = paste_target(
                canvas, target, target_mask, left, top, occupied_mask
            )
            placements.append(
                Placement(
                    class_id=asset.class_id,
                    class_name=asset.class_name,
                    target_path=asset.path,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    rotation=rotation,
                    scale=scale,
                    black_ratio=black_ratio,
                )
            )

        if placements:
            return canvas.convert("RGB"), clear_mask, placements, background_path

    raise RuntimeError(f"failed to place any target for synthetic image {index}")


def write_stats(output_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    stats_path = output_root / "placements.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    crop_root = args.crop_root.resolve()
    positive_root = args.positive_root.resolve()
    output_root = args.output.resolve()

    crop_assets = load_crop_assets(crop_root)
    background_paths = iter_images(positive_root)
    if not crop_assets:
        raise FileNotFoundError(f"no crop PNGs found in {crop_root}")
    if not background_paths:
        raise FileNotFoundError(f"no background images found in {positive_root}")
    if args.min_targets < 1 or args.max_targets < args.min_targets:
        raise ValueError("--min-targets/--max-targets are invalid")

    prepare_output(output_root, args.clean)
    write_data_yaml(output_root)
    seg_module = load_seg_module()

    rows: list[dict[str, object]] = []
    for index in range(1, args.num + 1):
        split = split_for_index(index, args.num, args.val_ratio)
        stem = f"synthetic_{index:05d}"

        image, clear_mask, placements, background_path = compose_one(
            index, background_paths, crop_assets, seg_module, rng, args
        )

        image_path = output_root / "images" / split / f"{stem}.jpg"
        label_path = output_root / "labels" / split / f"{stem}.txt"
        image.save(image_path, quality=95)
        label_path.write_text(
            "\n".join(yolo_line(p, image.width, image.height) for p in placements) + "\n",
            encoding="utf-8",
        )

        if index <= args.visualize_limit:
            preview = draw_preview(image, clear_mask, placements)
            preview.save(output_root / "visualize" / split / f"{stem}.jpg", quality=92)
            save_placement_mask(
                output_root / "placement_masks" / split / f"{stem}.png",
                clear_mask,
            )

        for placement_idx, placement in enumerate(placements, start=1):
            rows.append(
                {
                    "image": str(image_path.relative_to(output_root)),
                    "label": str(label_path.relative_to(output_root)),
                    "split": split,
                    "placement_index": placement_idx,
                    "background": str(background_path),
                    "target": str(placement.target_path),
                    "class_id": placement.class_id,
                    "class_name": placement.class_name,
                    "x1": placement.x1,
                    "y1": placement.y1,
                    "x2": placement.x2,
                    "y2": placement.y2,
                    "rotation": round(placement.rotation, 4),
                    "scale": round(placement.scale, 4),
                    "black_ratio": round(placement.black_ratio, 6),
                }
            )

        print(
            f"[{index:04d}/{args.num:04d}] {split}/{stem}.jpg "
            f"objects={len(placements)} bg={background_path.name}"
        )

    write_stats(output_root, rows)
    print(f"[DONE] dataset: {output_root}")
    print(f"[DONE] data yaml: {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
