# -*- coding: utf-8 -*-
"""
土地利用转移矩阵 2020 → 2025
"""

import numpy as np
from osgeo import gdal

# ==========================================
# 配置
# ==========================================
tif_2020 = r"D:\lunwen\Results_v7_2020\LULC_2020_final.tif"
tif_2025 = r"D:\lunwen\Results_v7_2025\LULC_2025_final.tif"
output_txt = r"D:\lunwen\TransferMatrix_2020_2025.txt"

PIXEL_AREA_M2 = 100  # 10m x 10m = 100 平方米

label_map = {1:"耕地", 2:"林地", 4:"草地", 5:"水域", 7:"裸地", 8:"建筑用地"}
classes = sorted(label_map.keys())

# ==========================================
# 读取
# ==========================================
ds_2020 = gdal.Open(tif_2020)
ds_2025 = gdal.Open(tif_2025)

arr_2020 = ds_2020.GetRasterBand(1).ReadAsArray()
arr_2025 = ds_2025.GetRasterBand(1).ReadAsArray()

ds_2020 = None
ds_2025 = None

# 只保留两年都有效的像素
valid_mask = (arr_2020 > 0) & (arr_2025 > 0)
a2020 = arr_2020[valid_mask]
a2025 = arr_2025[valid_mask]

# ==========================================
# 计算转移矩阵（像素数 → 平方米）
# ==========================================
n = len(classes)
matrix = np.zeros((n, n), dtype=np.float64)

for i, c_from in enumerate(classes):
    for j, c_to in enumerate(classes):
        count = np.sum((a2020 == c_from) & (a2025 == c_to))
        matrix[i, j] = count * PIXEL_AREA_M2

# ==========================================
# 输出
# ==========================================
labels = [label_map[c] for c in classes]
col_width = 12

header = f"{'2020→2025':<10}" + "".join(f"{l:>{col_width}}" for l in labels) + f"{'合计':>{col_width}}"
lines = [header, "-" * len(header)]

for i, row_label in enumerate(labels):
    row_total = matrix[i].sum()
    row_str = f"{row_label:<10}" + "".join(f"{matrix[i,j]:>{col_width}.0f}" for j in range(n)) + f"{row_total:>{col_width}.0f}"
    lines.append(row_str)

lines.append("-" * len(header))
col_totals = f"{'合计':<10}" + "".join(f"{matrix[:,j].sum():>{col_width}.0f}" for j in range(n)) + f"{matrix.sum():>{col_width}.0f}"
lines.append(col_totals)

result_text = "\n".join(lines)
print(result_text)

with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("土地利用转移矩阵 2020 → 2025（单位：平方米）\n\n")
    f.write(result_text)

print(f"\n已保存到: {output_txt}")