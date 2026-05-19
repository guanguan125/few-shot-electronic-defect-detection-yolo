# Few-Shot Electronic Defect Detection with YOLO

A YOLO-based defect detection pipeline for electronic product surface inspection.
The project supports six-class defect training, few-shot crop-based data
augmentation, synthetic image generation, model training, and batch inference.

## Project Structure

```text
configs/
  data_6_enhence.yaml          # YOLO dataset config for six-class enhanced data
scripts/
  train_6.py                   # Six-class training entrypoint
  inference.py                 # Batch inference and JSON export
  enhence/
    crop.py                    # Crop transparent defect targets from annotations
    colloect_positive_example.py
                               # Collect target-free background images
    concatenate_images.py       # Compose synthetic YOLO-format training data
first_data/
  train/
    正样本/                     # Clean positive/background images
    负样本/                     # Labeled defect images
  ps后的负样本/                 # Defect-removed background images
  enhence/                     # Generated augmentation assets and synthetic data
  test/image/                  # Test images for inference
  result/                      # Inference results
pretrained_weights/
  yolo26x.pt                   # Base pretrained YOLO weight
```

## Environment

The recommended CUDA version is 11.8.

```bash
conda create -n yolo python=3.10 -y
conda activate yolo
```

Install PyTorch with CUDA 11.8:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install Ultralytics:

```bash
pip install ultralytics==8.4.48
```

## Defect Classes

The training dataset uses six fine-grained classes:

```yaml
0: collision
1: dirt
2: plain_particle_dent
3: plain_particle_particle
4: scratch_damage
5: scratch_mark
```

During inference JSON export, the six classes are merged into four defect types:

```text
plain_particle_dent      -> plain particle
plain_particle_particle  -> plain particle
scratch_damage           -> scratch
scratch_mark             -> scratch
collision                -> collision
dirt                     -> dirt
```

## Synthetic Data Generation

Generate transparent defect crops from labeled negative samples:

```bash
python scripts/enhence/crop.py --clean
```

Collect target-free background images:

```bash
python scripts/enhence/colloect_positive_example.py --clean
```

Generate synthetic YOLO-format data with 1000 pasted targets per class:

```bash
python scripts/enhence/concatenate_images.py \
  --clean \
  --per-class-targets 1000 \
  --max-targets 3 \
  --visualize-limit 0
```

The generated dataset is saved to:

```text
first_data/enhence/concatenate_yolo/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
  placements.csv
```

Use `--num 1000` instead of `--per-class-targets 1000` if you want exactly
1000 synthetic images rather than 1000 targets per class.

## Training

Train the six-class detector:

```bash
python scripts/train_6.py
```

The current training script uses:

```text
weights: pretrained_weights/yolo26x.pt
data:    configs/data_6_enhence.yaml
output:  pretrained_weights/yolo26x_defect_6cls_full/
```

The best model is saved as:

```text
pretrained_weights/yolo26x_defect_6cls_full/weights/best.pt
```

## Inference

Run batch inference on all images under `first_data/test/image`:

```bash
python scripts/inference.py --clean
```

Default inference settings:

```text
weights: pretrained_weights/yolo26x_defect_6cls_full/weights/best.pt
input:   first_data/test/image
output:  first_data/result
```

Inference outputs:

```text
first_data/result/
  images/          # Side-by-side original image and bbox visualization
  labels/          # YOLO txt predictions
  json/            # Per-image JSON results with four merged defect labels
  detections.csv   # Detection summary table
```

## JSON Output Format

Each image produces one JSON file:

```json
{
  "image_id": "image_name.jpg",
  "annotations": [
    {
      "label": "scratch",
      "bbox": [10.5, 20.0, 100.2, 150.8],
      "confidence": 0.873421
    }
  ]
}
```

## Notes

- Run all commands from the project root:

```bash
cd /home/fhr/programs/projects/detection
conda activate yolo
```

- If GPU memory is insufficient during training, reduce `batch` or `imgsz` in
  `scripts/train_6.py`.
- If inference uses too much memory, run:

```bash
python scripts/inference.py --clean --batch 1 --imgsz 1024
```







第二种增强方法，图像预处理
/home/fhr/programs/projects/detection/scripts/prepross