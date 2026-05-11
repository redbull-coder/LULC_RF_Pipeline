# -*- coding: utf-8 -*-
"""
横沥镇 2020 年土地利用分类（sklearn 全9波段版）
- 使用全部9个GEE导出波段作为特征
- sklearn随机森林，建筑用地过采样
"""

import arcpy
import os
import numpy as np
from arcpy.sa import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

# ==========================================
# 配置区
# ==========================================
base_dir = r"D:\lunwen"
s2_path = os.path.join(base_dir, r"Sentine2_new\S2_Hengli_Stack_2020.tif")
clcd_raw = os.path.join(base_dir, r"CLCD\CLCD_v01_2020_albert_guangzhou.tif")
boundary = os.path.join(base_dir, r"hengli_441302110000\hengli_441302110000.shp")

results_dir = os.path.join(base_dir, "Results_Sklearn_9bands")
os.makedirs(results_dir, exist_ok=True)
out_tif = os.path.join(results_dir, "LULC_2020_9bands.tif")
accuracy_file = os.path.join(results_dir, "Accuracy_2020_9bands.txt")

TREES = 250                  # 增加树数量
SAMPLES_PER_CLASS = 400
VALIDATION_RATIO = 0.2
BUILTUP_CLASS = 8
OVERSAMPLE_FACTOR = 2

# ==========================================
# 环境设置
# ==========================================
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

# ==========================================
# 1. 对齐CLCD到S2
# ==========================================
print("[1/5] 对齐 CLCD 到 Sentinel-2 影像...")
s2_r = Raster(s2_path)
s2_sr = arcpy.Describe(s2_r).spatialReference
s2_extent = arcpy.Describe(s2_r).extent
cell_size = s2_r.meanCellWidth

arcpy.env.outputCoordinateSystem = s2_sr
arcpy.env.snapRaster = s2_path
arcpy.env.extent = s2_extent
arcpy.env.cellSize = cell_size

clcd_align = ExtractByMask(Raster(clcd_raw), boundary)
clcd_align_path = os.path.join(results_dir, "CLCD_align.tif")
clcd_align.save(clcd_align_path)

# ==========================================
# 2. 读取影像数据（全部9个波段）
# ==========================================
print("[2/5] 读取影像数据（9个波段）...")
clcd_arr = arcpy.RasterToNumPyArray(clcd_align_path, nodata_to_value=-9999)

# 提取全部9个波段（注意你的波段顺序）
band_list = []
for i in range(1, 10):
    band_arr = arcpy.RasterToNumPyArray(ExtractBand(s2_path, i))
    band_list.append(band_arr)

# 堆叠为 (rows, cols, 9)
features = np.stack(band_list, axis=-1).astype(np.float32)
# 处理无穷值和NaN
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

desc = arcpy.Describe(clcd_align_path)
xmin = desc.extent.XMin
ymax = desc.extent.YMax
cw = desc.meanCellWidth
ch = desc.meanCellHeight
rows, cols = clcd_arr.shape

print(f"  影像尺寸: {rows} 行 x {cols} 列, 特征数: 9")

# ==========================================
# 3. 分层抽样（含过采样）
# ==========================================
print("[3/5] 分层抽样与建筑用地过采样...")
unique_orig = np.unique(clcd_arr[clcd_arr != -9999])
unique_orig.sort()
orig_to_code = {orig: i for i, orig in enumerate(unique_orig)}
code_to_orig = {i: orig for orig, i in orig_to_code.items()}
print(f"  类别映射: {orig_to_code}")

X_train_list, y_train_list = [], []
X_valid_list, y_valid_list = [], []

for orig_cls in unique_orig:
    r_idx, c_idx = np.where(clcd_arr == orig_cls)
    total = len(r_idx)
    n_train = int(min(SAMPLES_PER_CLASS, total) * (1 - VALIDATION_RATIO))
    n_valid = min(SAMPLES_PER_CLASS, total) - n_train
    if n_train == 0:
        continue

    perm = np.random.permutation(total)
    train_idx = perm[:n_train]
    valid_idx = perm[n_train:n_train+n_valid]

    train_feat = features[r_idx[train_idx], c_idx[train_idx]]
    valid_feat = features[r_idx[valid_idx], c_idx[valid_idx]]

    code_val = orig_to_code[orig_cls]

    if orig_cls == BUILTUP_CLASS:
        train_feat = np.repeat(train_feat, OVERSAMPLE_FACTOR, axis=0)
        train_label = np.full(train_feat.shape[0], code_val)
    else:
        train_label = np.full(n_train, code_val)

    valid_label = np.full(n_valid, code_val)

    X_train_list.append(train_feat)
    y_train_list.append(train_label)
    X_valid_list.append(valid_feat)
    y_valid_list.append(valid_label)

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)
X_valid = np.vstack(X_valid_list)
y_valid = np.concatenate(y_valid_list)

print(f"  训练样本: {len(y_train)} 个, 验证样本: {len(y_valid)} 个")

# ==========================================
# 4. 训练随机森林
# ==========================================
print("[4/5] 训练随机森林（9波段）...")
rf = RandomForestClassifier(n_estimators=TREES, max_depth=30, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("  训练完成。")

# ==========================================
# 5. 全图预测
# ==========================================
print("[5/5] 全图预测并保存...")
flat_features = features.reshape(-1, 9)
chunk_size = 1024 * 1024
n_pixels = flat_features.shape[0]
pred_flat = np.zeros(n_pixels, dtype=np.int32)
for i in range(0, n_pixels, chunk_size):
    end = min(i + chunk_size, n_pixels)
    pred_flat[i:end] = rf.predict(flat_features[i:end])

pred_arr = pred_flat.reshape(rows, cols)

# 还原原始类别
final_arr = np.copy(pred_arr).astype(np.int16)
for code, orig in code_to_orig.items():
    final_arr[pred_arr == code] = orig

out_raster = arcpy.NumPyArrayToRaster(final_arr, arcpy.Point(xmin, ymax), cw, ch)
out_raster.save(out_tif)
print(f"  分类图已保存: {out_tif}")

# ==========================================
# 6. 精度评估
# ==========================================
y_pred = rf.predict(X_valid)
oa = accuracy_score(y_valid, y_pred)
kappa = cohen_kappa_score(y_valid, y_pred)
report = classification_report(y_valid, y_pred, zero_division=0)

with open(accuracy_file, 'w', encoding='utf-8') as f:
    f.write(f"总体精度 (OA): {oa:.4f}\n")
    f.write(f"Kappa 系数: {kappa:.4f}\n\n")
    f.write("分类报告:\n")
    f.write(report)

print(f"\n✅ 分类完成！")
print(f"OA = {oa:.4f}, Kappa = {kappa:.4f}")
print(f"精度报告: {accuracy_file}")
