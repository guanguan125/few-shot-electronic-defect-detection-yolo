import cv2
import numpy as np


img_path = "/home/fhr/programs/projects/detection/first_data/train/负样本/dirt/B22009S00_004_1_20220728135451799_000008_0_1.jpg"


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

def show_compare(name, original, processed, window_width=1200, window_height=700):
    if len(processed.shape) == 2:
        processed_show = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        processed_show = processed

    if len(original.shape) == 2:
        original_show = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        original_show = original

    concat = np.hstack([original_show, processed_show])

    # 按比例缩放到指定窗口大小内
    h, w = concat.shape[:2]
    scale = min(window_width / w, window_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    concat_resize = cv2.resize(concat, (new_w, new_h))

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, window_width, window_height)
    cv2.imshow(name, concat_resize)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"图像读取失败，请检查路径: {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ===============================
# 1. CLAHE 局部对比度增强
# ===============================
clahe = cv2.createCLAHE(
    clipLimit=2.5,
    tileGridSize=(8, 8)
)
clahe_img = clahe.apply(gray)
show_compare("Original | CLAHE", gray, clahe_img)


# ===============================
# 2. Gaussian High-pass 高频增强
# 原图 - 模糊图，适合颗粒、凹点、小缺陷
# ===============================
blur = cv2.GaussianBlur(gray, (31, 31), 0)
highpass = cv2.subtract(gray, blur)
highpass = normalize_to_uint8(highpass)
show_compare("Original | High-pass", gray, highpass)


# ===============================
# 3. Laplacian 二阶边缘增强
# 适合碰撞、划痕、局部结构变化
# ===============================
lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
lap = np.abs(lap)
lap = normalize_to_uint8(lap)
show_compare("Original | Laplacian", gray, lap)


# ===============================
# 4. Sobel 边缘增强
# ===============================
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = normalize_to_uint8(sobel)
show_compare("Original | Sobel", gray, sobel)


# ===============================
# 5. Scharr 细边缘增强
# 比 Sobel 更适合细划痕
# ===============================
scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
scharr = cv2.magnitude(scharrx, scharry)
scharr = normalize_to_uint8(scharr)
show_compare("Original | Scharr", gray, scharr)


# ===============================
# 6. White Top-hat
# 增强亮颗粒、亮划痕
# ===============================
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
tophat = normalize_to_uint8(tophat)
show_compare("Original | White Top-hat", gray, tophat)


# ===============================
# 7. Black-hat
# 增强暗凹坑、暗划痕、暗污渍
# ===============================
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
blackhat = normalize_to_uint8(blackhat)
show_compare("Original | Black-hat", gray, blackhat)


# ===============================
# 8. Gabor 方向纹理增强
# 适合划痕类缺陷
# ===============================
gabor_sum = np.zeros_like(gray, dtype=np.float32)

for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
    gabor_kernel = cv2.getGaborKernel(
        ksize=(31, 31),
        sigma=4.0,
        theta=theta,
        lambd=10.0,
        gamma=0.5,
        psi=0,
        ktype=cv2.CV_32F
    )

    filtered = cv2.filter2D(gray, cv2.CV_32F, gabor_kernel)
    gabor_sum = np.maximum(gabor_sum, filtered)

gabor = normalize_to_uint8(gabor_sum)
show_compare("Original | Gabor", gray, gabor)


# ===============================
# 9. Canny 边缘检测
# 适合明显边缘，但容易受纹理干扰
# ===============================
canny = cv2.Canny(gray, 50, 150)
show_compare("Original | Canny", gray, canny)


# ===============================
# 10. CLAHE + Scharr
# 推荐组合：增强局部对比度后检测细划痕
# ===============================
scharrx = cv2.Scharr(clahe_img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(clahe_img, cv2.CV_64F, 0, 1)
clahe_scharr = cv2.magnitude(scharrx, scharry)
clahe_scharr = normalize_to_uint8(clahe_scharr)
show_compare("Original | CLAHE + Scharr", gray, clahe_scharr)


# ===============================
# 11. CLAHE + Black-hat
# 推荐组合：增强暗缺陷
# ===============================
clahe_blackhat = cv2.morphologyEx(clahe_img, cv2.MORPH_BLACKHAT, kernel)
clahe_blackhat = normalize_to_uint8(clahe_blackhat)
show_compare("Original | CLAHE + Black-hat", gray, clahe_blackhat)


# ===============================
# 12. CLAHE + Top-hat
# 推荐组合：增强亮缺陷
# ===============================
clahe_tophat = cv2.morphologyEx(clahe_img, cv2.MORPH_TOPHAT, kernel)
clahe_tophat = normalize_to_uint8(clahe_tophat)
show_compare("Original | CLAHE + Top-hat", gray, clahe_tophat)


print("所有预处理方法显示完成。")