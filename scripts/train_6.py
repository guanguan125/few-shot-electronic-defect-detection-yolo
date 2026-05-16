from ultralytics import YOLO

def train():
    model = YOLO("/home/fhr/programs/projects/detection/pretrained_weights/yolo26x.pt")

    model.train(
        data="/home/fhr/programs/projects/detection/configs/data_6.yaml",
        epochs=120,
        imgsz=1536,
        batch=2,
        workers=8,
        device=0,

        project="/home/fhr/programs/projects/detection/pretrained_weights",
        name="yolo26x_defect_6cls",
        exist_ok=True,

        pretrained=True,
        freeze=10,          # 小数据微调建议冻结前面部分层
        patience=40,

        optimizer="AdamW",
        lr0=0.0005,         # 你现在的 0.01 对小数据偏大
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,

        augment=True,
        mosaic=0.2,         # 小数据可用，但不要太强
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,

        degrees=3.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,

        cache=True,
        amp=True
    )

if __name__ == "__main__":
    train()
