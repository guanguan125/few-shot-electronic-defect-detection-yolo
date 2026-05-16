from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "first_data" / "enhence" / "positive"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

SOURCES = {
    "train_positive": PROJECT_ROOT / "first_data" / "train" / "正样本",
    "ps_negative": PROJECT_ROOT / "first_data" / "ps后的负样本",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect target-free background images into one positive folder."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output positive folder before copying images.",
    )
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def output_name(source_key: str, source_root: Path, image_path: Path) -> str:
    rel = image_path.relative_to(source_root)
    stem_parts = [source_key, *rel.with_suffix("").parts]
    safe_stem = "__".join(part.replace(" ", "_") for part in stem_parts)
    return f"{safe_stem}{image_path.suffix.lower()}"


def main() -> None:
    args = parse_args()
    output_root = args.output.resolve()

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copied = Counter()
    skipped = Counter()
    used_names: set[str] = set()

    for source_key, source_root in SOURCES.items():
        source_root = source_root.resolve()
        images = iter_images(source_root)
        if not images:
            skipped[f"{source_key}_empty_or_missing"] += 1
            continue

        for image_path in images:
            name = output_name(source_key, source_root, image_path)
            if name in used_names:
                skipped["name_collision"] += 1
                continue
            shutil.copy2(image_path, output_root / name)
            used_names.add(name)
            copied[source_key] += 1

    print(f"output: {output_root}")
    print("copied:")
    for source_key in SOURCES:
        print(f"  {source_key}: {copied[source_key]}")
    if skipped:
        print("skipped:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
