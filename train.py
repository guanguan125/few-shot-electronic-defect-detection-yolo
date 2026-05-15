from ultralytics import YOLO

def train():
    # 使用官方预训练模型（推荐）
    model = YOLO("/home/fhr/programs/projects/detection/pretrained_weights/yolo26x.pt")  # 可换 yolov8n.pt / yolov8m.pt

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,
        device=0,  # GPU
        project="defect_yolo",
        name="exp1",
        patience=20,  # early stop
        lr0=0.01,
        augment=True
    )

if __name__ == "__main__":
    train()