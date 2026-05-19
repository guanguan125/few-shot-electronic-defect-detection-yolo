from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path("/home/fhr/programs/projects/detection")
DEFAULT_SOURCE = PROJECT_ROOT / "first_data" / "test" / "image"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a batch of images for manual labeling while skipping images "
            "that already exist in the output folder."
        )
    )
    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help="Number of images to copy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder for selected images.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source image folder. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible selection.",
    )
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="Select images in filename order instead of random order.",
    )
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def existing_image_names(output: Path) -> set[str]:
    if not output.exists():
        return set()
    return {
        path.name
        for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if args.num <= 0:
        raise ValueError("--num must be greater than 0")
    if not source.exists():
        raise FileNotFoundError(f"source folder not found: {source}")

    output.mkdir(parents=True, exist_ok=True)
    images = iter_images(source)
    used_names = existing_image_names(output)
    candidates = [path for path in images if path.name not in used_names]

    if not candidates:
        raise RuntimeError(f"no selectable images left. output already has all images: {output}")

    if args.ordered:
        selected = candidates[: args.num]
    else:
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        selected = candidates[: args.num]

    for image_path in selected:
        shutil.copy2(image_path, output / image_path.name)

    print(f"source: {source}")
    print(f"output: {output}")
    print(f"source_images: {len(images)}")
    print(f"already_in_output: {len(used_names)}")
    print(f"selectable: {len(candidates)}")
    print(f"copied: {len(selected)}")
    if len(selected) < args.num:
        print(f"warning: requested {args.num}, but only {len(selected)} images were available")


if __name__ == "__main__":
    main()
