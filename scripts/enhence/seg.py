#!/usr/bin/env python3
"""
Segment PS negative samples into black, white and fuzzy regions.

Default idea:
1. Convert each image to grayscale and lightly smooth JPEG noise.
2. Use Otsu to find the black/white split for the current image.
3. Keep a small uncertain gray band around that split as "fuzzy".
4. Add a narrow morphology boundary band around black/white transitions.
5. Detect defocused regions with local high-frequency energy and mark them
   as "fuzzy" too.

Outputs:
- label/*.png: single-channel labels, 0=black, 1=white, 2=fuzzy
- mask_black/*.png, mask_white/*.png, mask_fuzzy/*.png: binary masks
- preview/*.jpg: original + color label + overlay
- stats.csv: thresholds and area ratios
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_INPUT_DIR = Path(
    "/home/fhr/programs/projects/detection/first_data/ps后的负样本"
)
DEFAULT_OUTPUT_DIR = Path("/home/fhr/programs/projects/detection/seg/output")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

LABEL_BLACK = 0
LABEL_WHITE = 1
LABEL_FUZZY = 2


@dataclass
class SegmentResult:
    label: np.ndarray
    black_mask: np.ndarray
    white_mask: np.ndarray
    fuzzy_mask: np.ndarray
    defocus_mask: np.ndarray
    threshold_low: int
    threshold_high: int
    threshold_main: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment grayscale PS negative samples into black/white/fuzzy."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input image directory. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--method",
        choices=("adaptive", "multiotsu"),
        default="adaptive",
        help=(
            "adaptive: Otsu black/white split plus uncertain transition band. "
            "multiotsu: direct 3-level gray segmentation."
        ),
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        help="Odd Gaussian blur kernel for noise suppression. Use 1 to disable.",
    )
    parser.add_argument(
        "--uncertainty-ratio",
        type=float,
        default=0.08,
        help=(
            "Width of the fuzzy gray band around the adaptive threshold, "
            "as a ratio of robust image contrast."
        ),
    )
    parser.add_argument(
        "--min-margin",
        type=int,
        default=8,
        help="Minimum gray-level half width for the fuzzy threshold band.",
    )
    parser.add_argument(
        "--max-margin",
        type=int,
        default=32,
        help="Maximum gray-level half width for the fuzzy threshold band.",
    )
    parser.add_argument(
        "--transition-radius",
        type=int,
        default=2,
        help=(
            "Pixels around black/white boundaries also marked as fuzzy. "
            "Use 0 to disable boundary expansion."
        ),
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=0,
        help=(
            "Move black/white connected components smaller than this many pixels "
            "into fuzzy. Default 0 keeps all details."
        ),
    )
    parser.add_argument(
        "--no-defocus",
        action="store_true",
        help="Disable local blur / defocus detection.",
    )
    parser.add_argument(
        "--defocus-window",
        type=int,
        default=65,
        help="Local window size for focus measurement.",
    )
    parser.add_argument(
        "--defocus-min-contrast",
        type=float,
        default=8.0,
        help=(
            "Minimum local standard deviation used as evidence-bearing area. "
            "This avoids calling flat saturated black/white areas blurry."
        ),
    )
    parser.add_argument(
        "--defocus-percentile",
        type=float,
        default=40.0,
        help="Percentile of local high-frequency energy treated as defocused.",
    )
    parser.add_argument(
        "--defocus-max-score",
        type=float,
        default=2.8,
        help=(
            "Absolute upper limit for defocus high-frequency score. "
            "Use 0 to disable this cap."
        ),
    )
    parser.add_argument(
        "--defocus-close",
        type=int,
        default=51,
        help="Morphological close kernel for joining defocus regions.",
    )
    parser.add_argument(
        "--defocus-open",
        type=int,
        default=31,
        help="Morphological open kernel for removing tiny defocus speckles.",
    )
    parser.add_argument(
        "--defocus-min-area-ratio",
        type=float,
        default=0.005,
        help="Minimum connected defocus component area ratio to keep.",
    )
    parser.add_argument(
        "--defocus-row-fill-ratio",
        type=float,
        default=0.18,
        help=(
            "If a row has this ratio of defocus evidence, fill the whole row. "
            "This catches large horizontal out-of-focus bands."
        ),
    )
    parser.add_argument(
        "--defocus-min-row-run",
        type=float,
        default=0.08,
        help="Minimum image-height ratio for a row-filled defocus band.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N images. Default 0 means all images.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not save preview images.",
    )
    return parser.parse_args()


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def normalize_odd_kernel(value: int) -> int:
    if value <= 1:
        return 1
    if value % 2 == 0:
        value += 1
    return max(value, 3)


def smooth_gray(gray: np.ndarray, blur_kernel: int) -> np.ndarray:
    kernel = normalize_odd_kernel(blur_kernel)
    if kernel <= 1:
        return gray
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def multi_otsu_3(gray: np.ndarray) -> tuple[int, int]:
    """Find two thresholds that maximize 3-class Otsu between-class variance."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 85, 170

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256, dtype=np.float64))
    mu_total = mu[-1]

    best_score = -1.0
    best_t1 = 85
    best_t2 = 170
    eps = 1e-12

    for t1 in range(1, 254):
        w0 = omega[t1]
        if w0 <= eps:
            continue

        w1 = omega[t1 + 1 : 255] - w0
        w2 = 1.0 - omega[t1 + 1 : 255]
        valid = (w1 > eps) & (w2 > eps)
        if not valid.any():
            continue

        m0 = mu[t1] / w0
        m1 = np.zeros_like(w1)
        m2 = np.zeros_like(w2)
        m1[valid] = (mu[t1 + 1 : 255][valid] - mu[t1]) / w1[valid]
        m2[valid] = (mu_total - mu[t1 + 1 : 255][valid]) / w2[valid]

        scores = (
            w0 * (m0 - mu_total) ** 2
            + w1 * (m1 - mu_total) ** 2
            + w2 * (m2 - mu_total) ** 2
        )
        scores[~valid] = -1.0
        local_idx = int(np.argmax(scores))
        local_score = float(scores[local_idx])

        if local_score > best_score:
            best_score = local_score
            best_t1 = t1
            best_t2 = t1 + 1 + local_idx

    return int(best_t1), int(best_t2)


def otsu_threshold(gray: np.ndarray) -> int:
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(round(threshold))


def remove_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, np.ndarray]:
    if min_area <= 0:
        empty = np.zeros_like(mask, dtype=bool)
        return mask, empty

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros_like(mask, dtype=bool)
    removed = np.zeros_like(mask, dtype=bool)

    for idx in range(1, num_labels):
        component = labels == idx
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            keep |= component
        else:
            removed |= component

    return keep, removed


def keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros_like(mask, dtype=bool)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == idx
    return keep


def fill_row_runs(mask: np.ndarray, row_fill_ratio: float, min_row_run: float) -> np.ndarray:
    if row_fill_ratio <= 0:
        return mask

    row_smooth = max(3, normalize_odd_kernel(int(mask.shape[0] * 0.04)))
    row_density = cv2.blur(mask.astype(np.float32), (1, row_smooth)).mean(axis=1)
    active_rows = row_density >= row_fill_ratio
    min_rows = max(1, int(round(mask.shape[0] * min_row_run)))

    filled = mask.copy()
    start: int | None = None
    rows_with_sentinel = list(active_rows) + [False]
    for row_idx, is_active in enumerate(rows_with_sentinel):
        if is_active and start is None:
            start = row_idx
        elif not is_active and start is not None:
            if row_idx - start >= min_rows:
                filled[start:row_idx, :] = True
            start = None

    return filled


def build_defocus_mask(gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.no_defocus:
        return np.zeros_like(gray, dtype=bool)

    window = normalize_odd_kernel(args.defocus_window)
    close_size = normalize_odd_kernel(args.defocus_close)
    open_size = normalize_odd_kernel(args.defocus_open)

    gray_f = gray.astype(np.float32)
    local_mean = cv2.boxFilter(gray_f, -1, (window, window), normalize=True)
    local_mean2 = cv2.boxFilter(gray_f * gray_f, -1, (window, window), normalize=True)
    local_std = np.sqrt(np.maximum(local_mean2 - local_mean * local_mean, 0))

    fine = cv2.GaussianBlur(gray_f, (0, 0), 1.0)
    coarse = cv2.GaussianBlur(gray_f, (0, 0), 5.0)
    high_freq = np.abs(fine - coarse)
    local_high_freq = cv2.boxFilter(
        high_freq, -1, (window, window), normalize=True
    )

    support = local_std >= args.defocus_min_contrast
    if support.mean() < 0.02:
        return np.zeros_like(gray, dtype=bool)

    score_threshold = float(
        np.percentile(local_high_freq[support], args.defocus_percentile)
    )
    if args.defocus_max_score > 0:
        score_threshold = min(score_threshold, args.defocus_max_score)

    seed = support & (local_high_freq <= score_threshold)
    seed_u8 = seed.astype(np.uint8)

    if close_size > 1:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )
        seed_u8 = cv2.morphologyEx(seed_u8, cv2.MORPH_CLOSE, close_kernel)

    if open_size > 1:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        seed_u8 = cv2.morphologyEx(seed_u8, cv2.MORPH_OPEN, open_kernel)

    min_area = max(1000, int(round(gray.size * args.defocus_min_area_ratio)))
    defocus_mask = keep_large_components(seed_u8.astype(bool), min_area)
    defocus_mask = fill_row_runs(
        defocus_mask, args.defocus_row_fill_ratio, args.defocus_min_row_run
    )

    return defocus_mask


def apply_fuzzy_override(
    black_mask: np.ndarray,
    white_mask: np.ndarray,
    fuzzy_mask: np.ndarray,
    extra_fuzzy_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fuzzy_mask = fuzzy_mask | extra_fuzzy_mask
    black_mask = black_mask & ~fuzzy_mask
    white_mask = white_mask & ~fuzzy_mask
    return black_mask, white_mask, fuzzy_mask


def add_transition_band(
    black_mask: np.ndarray,
    white_mask: np.ndarray,
    fuzzy_mask: np.ndarray,
    threshold_basis: np.ndarray,
    threshold_main: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radius <= 0:
        return black_mask, white_mask, fuzzy_mask

    hard = (threshold_basis >= threshold_main).astype(np.uint8)
    boundary = cv2.morphologyEx(
        hard,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    if radius > 1:
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        boundary = cv2.dilate(boundary, kernel, iterations=1)

    transition = boundary.astype(bool)
    fuzzy_mask = fuzzy_mask | transition
    black_mask = black_mask & ~fuzzy_mask
    white_mask = white_mask & ~fuzzy_mask
    return black_mask, white_mask, fuzzy_mask


def build_label(
    black_mask: np.ndarray, white_mask: np.ndarray, fuzzy_mask: np.ndarray
) -> np.ndarray:
    label = np.full(black_mask.shape, LABEL_FUZZY, dtype=np.uint8)
    label[black_mask] = LABEL_BLACK
    label[white_mask] = LABEL_WHITE
    label[fuzzy_mask] = LABEL_FUZZY
    return label


def segment_adaptive(gray: np.ndarray, args: argparse.Namespace) -> SegmentResult:
    basis = smooth_gray(gray, args.blur_kernel)
    threshold = otsu_threshold(basis)
    p_low, p_high = np.percentile(basis, [5, 95])
    contrast = max(1.0, float(p_high - p_low))
    margin = int(round(np.clip(contrast * args.uncertainty_ratio, args.min_margin, args.max_margin)))

    threshold_low = max(0, threshold - margin)
    threshold_high = min(255, threshold + margin)

    black_mask = basis <= threshold_low
    white_mask = basis >= threshold_high
    fuzzy_mask = ~(black_mask | white_mask)

    black_mask, white_mask, fuzzy_mask = add_transition_band(
        black_mask,
        white_mask,
        fuzzy_mask,
        basis,
        threshold,
        args.transition_radius,
    )

    black_mask, black_removed = remove_small_components(black_mask, args.min_area)
    white_mask, white_removed = remove_small_components(white_mask, args.min_area)
    fuzzy_mask = fuzzy_mask | black_removed | white_removed
    defocus_mask = build_defocus_mask(gray, args)
    black_mask, white_mask, fuzzy_mask = apply_fuzzy_override(
        black_mask, white_mask, fuzzy_mask, defocus_mask
    )

    label = build_label(black_mask, white_mask, fuzzy_mask)
    return SegmentResult(
        label=label,
        black_mask=black_mask,
        white_mask=white_mask,
        fuzzy_mask=fuzzy_mask,
        defocus_mask=defocus_mask,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        threshold_main=threshold,
    )


def segment_multiotsu(gray: np.ndarray, args: argparse.Namespace) -> SegmentResult:
    basis = smooth_gray(gray, args.blur_kernel)
    threshold_low, threshold_high = multi_otsu_3(basis)
    threshold_main = int(round((threshold_low + threshold_high) / 2))

    black_mask = basis <= threshold_low
    white_mask = basis >= threshold_high
    fuzzy_mask = (basis > threshold_low) & (basis < threshold_high)

    black_mask, white_mask, fuzzy_mask = add_transition_band(
        black_mask,
        white_mask,
        fuzzy_mask,
        basis,
        threshold_main,
        args.transition_radius,
    )

    black_mask, black_removed = remove_small_components(black_mask, args.min_area)
    white_mask, white_removed = remove_small_components(white_mask, args.min_area)
    fuzzy_mask = fuzzy_mask | black_removed | white_removed
    defocus_mask = build_defocus_mask(gray, args)
    black_mask, white_mask, fuzzy_mask = apply_fuzzy_override(
        black_mask, white_mask, fuzzy_mask, defocus_mask
    )

    label = build_label(black_mask, white_mask, fuzzy_mask)
    return SegmentResult(
        label=label,
        black_mask=black_mask,
        white_mask=white_mask,
        fuzzy_mask=fuzzy_mask,
        defocus_mask=defocus_mask,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        threshold_main=threshold_main,
    )


def colorize_label(label: np.ndarray) -> np.ndarray:
    color = np.zeros((*label.shape, 3), dtype=np.uint8)
    color[label == LABEL_BLACK] = (40, 40, 40)
    color[label == LABEL_WHITE] = (255, 255, 255)
    color[label == LABEL_FUZZY] = (0, 0, 255)
    return color


def add_legend(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    items = [
        ("black", (40, 40, 40)),
        ("white", (255, 255, 255)),
        ("fuzzy", (0, 0, 255)),
    ]
    x = 12
    y = 24
    for text, color in items:
        cv2.rectangle(result, (x, y - 14), (x + 22, y + 4), color, -1)
        cv2.rectangle(result, (x, y - 14), (x + 22, y + 4), (0, 0, 0), 1)
        cv2.putText(
            result,
            text,
            (x + 30, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        x += 118
    return result


def make_preview(gray: np.ndarray, label: np.ndarray) -> np.ndarray:
    original = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color_label = colorize_label(label)
    overlay = cv2.addWeighted(original, 0.58, color_label, 0.42, 0)
    preview = np.hstack([original, color_label, overlay])
    return add_legend(preview)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"failed to write image: {path}")


def relative_output_path(output_root: Path, folder: str, rel_path: Path, suffix: str) -> Path:
    return output_root / folder / rel_path.with_suffix(suffix)


def process_image(path: Path, input_root: Path, args: argparse.Namespace) -> dict[str, object]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise OSError(f"failed to read image: {path}")

    if args.method == "adaptive":
        result = segment_adaptive(gray, args)
    else:
        result = segment_multiotsu(gray, args)

    rel_path = path.relative_to(input_root)
    output_root = args.output

    write_image(relative_output_path(output_root, "label", rel_path, ".png"), result.label)
    write_image(
        relative_output_path(output_root, "mask_black", rel_path, ".png"),
        (result.black_mask.astype(np.uint8) * 255),
    )
    write_image(
        relative_output_path(output_root, "mask_white", rel_path, ".png"),
        (result.white_mask.astype(np.uint8) * 255),
    )
    write_image(
        relative_output_path(output_root, "mask_fuzzy", rel_path, ".png"),
        (result.fuzzy_mask.astype(np.uint8) * 255),
    )

    if not args.no_preview:
        preview = make_preview(gray, result.label)
        write_image(relative_output_path(output_root, "preview", rel_path, ".jpg"), preview)

    total = float(gray.size)
    return {
        "image": str(rel_path),
        "height": int(gray.shape[0]),
        "width": int(gray.shape[1]),
        "method": args.method,
        "threshold_low": result.threshold_low,
        "threshold_high": result.threshold_high,
        "threshold_main": result.threshold_main,
        "black_pixels": int(result.black_mask.sum()),
        "white_pixels": int(result.white_mask.sum()),
        "fuzzy_pixels": int(result.fuzzy_mask.sum()),
        "defocus_pixels": int(result.defocus_mask.sum()),
        "black_ratio": round(float(result.black_mask.sum()) / total, 6),
        "white_ratio": round(float(result.white_mask.sum()) / total, 6),
        "fuzzy_ratio": round(float(result.fuzzy_mask.sum()) / total, 6),
        "defocus_ratio": round(float(result.defocus_mask.sum()) / total, 6),
    }


def write_stats(output_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    stats_path = output_root / "stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    args.output = args.output.resolve()

    images = list(iter_images(input_root))
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        raise FileNotFoundError(f"no images found in {input_root}")

    rows: list[dict[str, object]] = []
    for idx, path in enumerate(images, start=1):
        row = process_image(path, input_root, args)
        rows.append(row)
        print(
            f"[{idx:04d}/{len(images):04d}] {row['image']} "
            f"black={row['black_ratio']:.3f} "
            f"white={row['white_ratio']:.3f} "
            f"fuzzy={row['fuzzy_ratio']:.3f} "
            f"thr=({row['threshold_low']},{row['threshold_high']})"
        )

    write_stats(args.output, rows)
    print(f"[DONE] output saved to: {args.output}")


if __name__ == "__main__":
    main()
