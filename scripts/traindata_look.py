import os
import json
import cv2
import numpy as np

# ======================
# 路径配置
# ======================
json_dir = "/home/fhr/programs/projects/detection/first_data/train/负样本"
ps_img_dir = "/home/fhr/programs/projects/detection/first_data/ps后的负样本"
save_dir = "/home/fhr/programs/projects/detection/first_data/look_ps"

os.makedirs(save_dir, exist_ok=True)

# ======================
# 遍历所有json
# ======================
for root, dirs, files in os.walk(json_dir):
    for file in files:
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(root, file)
        img_path = json_path.replace(".json", ".jpg")
        relative_dir = os.path.relpath(root, json_dir)
        ps_img_path = os.path.join(
            ps_img_dir,
            relative_dir,
            file.replace(".json", ".jpg")
        )

        if not os.path.exists(img_path):
            print(f"[WARN] image not found: {img_path}")
            continue

        if not os.path.exists(ps_img_path):
            print(f"[WARN] ps image not found: {ps_img_path}")
            continue

        # 读取原图
        img = cv2.imread(img_path)
        ps_img = cv2.imread(ps_img_path)
        img_vis = img.copy()

        # 读取json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])

        for shape in shapes:
            label = shape.get("label", "unknown")
            points = shape.get("points", [])

            if len(points) < 2:
                continue

            pts = np.array(points, dtype=np.int32)

            # 画轮廓
            cv2.polylines(img_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # 填充（半透明）
            overlay = img_vis.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
            img_vis = cv2.addWeighted(overlay, 0.2, img_vis, 0.8, 0)

            # 标注类别
            x, y = pts[0]
            cv2.putText(
                img_vis,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # ======================
        # 拼接：原图 + 标注图 + ps后图片
        # ======================
        h1, w1 = img.shape[:2]
        h2, w2 = img_vis.shape[:2]
        h3, w3 = ps_img.shape[:2]

        # 保证尺寸一致（安全处理）
        if h1 != h2 or w1 != w2:
            img_vis = cv2.resize(img_vis, (w1, h1))

        if h1 != h3 or w1 != w3:
            ps_img = cv2.resize(ps_img, (w1, h1))

        concat = np.hstack((img, img_vis, ps_img))

        # ======================
        # 保存
        # ======================
        if relative_dir == ".":
            target_dir = save_dir
        else:
            target_dir = os.path.join(save_dir, relative_dir)

        os.makedirs(target_dir, exist_ok=True)

        save_path = os.path.join(target_dir, file.replace(".json", "_compare.jpg"))
        cv2.imwrite(save_path, concat)

        print(f"[OK] saved: {save_path}")
