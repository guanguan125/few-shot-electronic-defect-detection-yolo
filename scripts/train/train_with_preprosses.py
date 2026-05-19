from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepross.yolo_preprocess_dataset import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    build_dataset,
)


DEFAULT_WEIGHTS = PROJECT_ROOT / "pretrained_weights" / "yolo26x.pt"
DEFAULT_PROJECT = PROJECT_ROOT / "pretrained_weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 4-channel CV-preprocessed grayscale defect data and train "
            "a YOLO detector with matching input channels."
        )
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="yolo26x_defect_4cls_preprocess")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild the preprocessed YOLO dataset before training.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only convert data and write yaml, then exit without training.",
    )
    return parser.parse_args()


def dataset_ready(output_root: Path, data_yaml: Path) -> bool:
    image_root = output_root / "images"
    label_root = output_root / "labels"
    if not data_yaml.exists() or not image_root.exists() or not label_root.exists():
        return False
    image_files = sorted(image_root.rglob("*.jpg"))
    if not image_files:
        return False
    for image_path in image_files:
        split = image_path.parent.name
        label_path = label_root / split / f"{image_path.stem}.txt"
        if not image_path.with_suffix(".npy").exists() or not label_path.exists():
            return False
    return True


def prepare_dataset(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.rebuild_data and dataset_ready(args.output, args.data):
        return None
    return build_dataset(
        source_root=args.source,
        output_root=args.output,
        config_path=args.data,
        val_ratio=args.val_ratio,
        seed=args.seed,
        clean=args.rebuild_data,
    )


def train(args: argparse.Namespace) -> None:
    summary = prepare_dataset(args)
    if summary:
        print(
            "prepared dataset: "
            f"{summary['images']} images, {summary['objects']} objects, "
            f"train={summary['train_images']}, val={summary['val_images']}"
        )

    if args.prepare_only:
        print(f"data yaml: {args.data}")
        return

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        freeze=0,
        patience=60,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=5,
        cos_lr=True,
        augment=True,
        mosaic=0.2,
        close_mosaic=20,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.4,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        cache=False,
        amp=True,
    )


if __name__ == "__main__":
    train(parse_args())
