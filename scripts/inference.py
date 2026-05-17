from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path("/home/fhr/programs/projects/detection")
DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "pretrained_weights"
    / "yolo26x_defect_6cls_full1"
    / "weights"
    / "best.pt"
)
DEFAULT_SOURCE = PROJECT_ROOT / "first_data" / "test" / "image"
DEFAULT_OUTPUT = PROJECT_ROOT / "first_data" / "result"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FOUR_CLASS_NAMES = {
    0: "collision",
    1: "dirt",
    2: "plain particle",
    3: "plain particle",
    4: "scratch",
    5: "scratch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO defect inference.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before running inference.",
    )
    return parser.parse_args()


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def prepare_output(output: Path, clean: bool) -> tuple[Path, Path, Path]:
    if clean and output.exists():
        shutil.rmtree(output)
    image_dir = output / "images"
    label_dir = output / "labels"
    json_dir = output / "json"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_dir, json_dir


def write_label(path: Path, result) -> None:
    lines: list[str] = []
    if result.boxes is not None and len(result.boxes) > 0:
        xywhn = result.boxes.xywhn.cpu().tolist()
        cls = result.boxes.cls.cpu().tolist()
        conf = result.boxes.conf.cpu().tolist()
        for class_id, score, box in zip(cls, conf, xywhn):
            x, y, w, h = box
            lines.append(
                f"{int(class_id)} {score:.6f} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def annotations_for_json(result) -> list[dict[str, object]]:
    annotations: list[dict[str, object]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return annotations

    xyxy = result.boxes.xyxy.cpu().tolist()
    cls = result.boxes.cls.cpu().tolist()
    conf = result.boxes.conf.cpu().tolist()
    for class_id, score, box in zip(cls, conf, xyxy):
        cid = int(class_id)
        x1, y1, x2, y2 = box
        annotations.append(
            {
                "label": FOUR_CLASS_NAMES.get(cid, str(cid)),
                "bbox": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2),
                ],
                "confidence": round(float(score), 6),
            }
        )
    return annotations


def write_result_json(path: Path, image_name: str, result) -> None:
    data = {
        "image_id": image_name,
        "annotations": annotations_for_json(result),
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_side_by_side_image(path: Path, result) -> None:
    original = result.orig_img
    plotted = result.plot()
    if original is None:
        raise OSError(f"missing original image for result: {result.path}")

    if original.shape[:2] != plotted.shape[:2]:
        plotted = cv2.resize(plotted, (original.shape[1], original.shape[0]))

    combined = np.concatenate([original, plotted], axis=1)
    ok = cv2.imwrite(str(path), combined)
    if not ok:
        raise OSError(f"failed to write image: {path}")


def result_rows(result, source_root: Path, names: dict[int, str]) -> list[dict[str, object]]:
    image_path = Path(result.path)
    try:
        rel_image = image_path.relative_to(source_root)
    except ValueError:
        rel_image = image_path.name

    rows: list[dict[str, object]] = []
    if result.boxes is None or len(result.boxes) == 0:
        rows.append(
            {
                "image": str(rel_image),
                "class_id": "",
                "class_name": "",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
            }
        )
        return rows

    xyxy = result.boxes.xyxy.cpu().tolist()
    cls = result.boxes.cls.cpu().tolist()
    conf = result.boxes.conf.cpu().tolist()
    for class_id, score, box in zip(cls, conf, xyxy):
        cid = int(class_id)
        x1, y1, x2, y2 = box
        rows.append(
            {
                "image": str(rel_image),
                "class_id": cid,
                "class_name": names.get(cid, str(cid)),
                "confidence": round(float(score), 6),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "image",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    source = args.source.resolve()
    output = args.output.resolve()

    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")
    images = iter_images(source)
    if not images:
        raise FileNotFoundError(f"no images found: {source}")

    image_dir, label_dir, json_dir = prepare_output(output, args.clean)
    model = YOLO(str(weights))
    names = model.names
    rows: list[dict[str, object]] = []

    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
    )

    for index, result in enumerate(results, start=1):
        image_path = Path(result.path)
        stem = image_path.stem

        result_image_path = image_dir / f"{stem}.jpg"
        save_side_by_side_image(result_image_path, result)

        label_path = label_dir / f"{stem}.txt"
        write_label(label_path, result)

        json_path = json_dir / f"{stem}.json"
        write_result_json(json_path, image_path.name, result)

        rows.extend(result_rows(result, source, names))

        count = 0 if result.boxes is None else len(result.boxes)
        print(f"[{index:04d}/{len(images):04d}] {image_path.name}: {count} detections")

    write_csv(output / "detections.csv", rows)
    print(f"[DONE] images: {image_dir}")
    print(f"[DONE] labels: {label_dir}")
    print(f"[DONE] json: {json_dir}")
    print(f"[DONE] csv: {output / 'detections.csv'}")


if __name__ == "__main__":
    main()
