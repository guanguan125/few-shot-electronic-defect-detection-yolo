# 黑 / 白 / 模糊 三分类分割

脚本位置：`seg.py`

默认输入：

```bash
/home/fhr/programs/projects/detection/first_data/ps后的负样本
```

默认输出：

```bash
/home/fhr/programs/projects/detection/seg/output
```

## 方法

默认使用 `adaptive` 方法：

1. 读入灰度图，并用轻微高斯滤波降低 JPEG 噪声。
2. 对每张图单独做 Otsu 自适应阈值，得到黑 / 白主分界。
3. 阈值附近的一段灰度不确定区标为 `fuzzy`。
4. 黑白交界处再用形态学膨胀出一圈过渡带，也标为 `fuzzy`。
5. 额外计算局部高频能量，检测失焦 / 模糊区域，并覆盖为 `fuzzy`。

这适合当前数据里亮度变化比较大的情况：不同图片的黑色主体不一定接近 0，白色区域也不一定只在 255，所以阈值需要按图自适应。

其中第 5 步是为了处理“亮度还能分黑白，但图像本身已经虚掉”的区域。例如
`dirt/B22009S00_004_2_20240201141256231_000004_0_2_1.jpg` 的下半部分会被整体标为模糊。

## 运行

```bash
python3 seg/seg.py
```

只试跑前 5 张：

```bash
python3 seg/seg.py --limit 5
```

如果希望把所有中间灰度直接视为模糊层，可以用三类 Otsu：

```bash
python3 seg/seg.py --method multiotsu
```

如果觉得红色模糊边缘太宽，可以减小过渡半径：

```bash
python3 seg/seg.py --transition-radius 1
```

如果觉得模糊灰度带太宽，可以减小不确定比例：

```bash
python3 seg/seg.py --uncertainty-ratio 0.04
```

如果觉得大面积失焦区域标得太多，可以提高局部对比门槛或降低失焦分位数：

```bash
python3 seg/seg.py --defocus-min-contrast 12 --defocus-percentile 30
```

如果只想保留旧版“阈值附近 + 边界过渡”的模糊定义，可以关闭失焦检测：

```bash
python3 seg/seg.py --no-defocus
```

## 输出含义

- `label/`：单通道类别图，像素值 `0=black`、`1=white`、`2=fuzzy`
- `mask_black/`：黑色部分二值 mask
- `mask_white/`：白色部分二值 mask
- `mask_fuzzy/`：模糊部分二值 mask
- `preview/`：原图、三分类结果、叠加图的横向对照
- `stats.csv`：每张图的阈值、黑/白/模糊/失焦像素比例
