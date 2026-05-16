from ultralytics import YOLO

# 三选一
model = YOLO("/home/fhr/programs/projects/detection/pretrained_weights/yolo26x.pt")  # 最快

results = model("/home/fhr/programs/projects/detection/first_data/train/负样本/plain particle/凹点/B22009S00_004_1_20220920163235874_000003_0_0.jpg", conf=0.25)

results[0].show()   # 弹窗显示