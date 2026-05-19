from ultralytics import YOLO

def train():
    model = YOLO("/home/fhr/programs/projects/detection/pretrained_weights/yolo26x.pt")

    model.train(
        data="/home/fhr/programs/projects/detection/configs/data_6_enhence.yaml",
        epochs=120,
        imgsz=1536,
        batch=1,
        workers=1,
        device=0,

        project="/home/fhr/programs/projects/detection/pretrained_weights",
        name="yolo26x_defect_6cls_full2",
        exist_ok=True,

        pretrained=True,
        freeze=0,           # 全参训练，不冻结 backbone
        patience=60,

        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=5,
        cos_lr=True,

        augment=True,
        mosaic=0.5,
        close_mosaic=20,
        mixup=0.05,
        copy_paste=0.0,

        degrees=5.0,
        translate=0.08,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,

        cache=False,
        amp=True
    )

if __name__ == "__main__":
    train()
