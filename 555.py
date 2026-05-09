# -*- coding: utf-8 -*-
"""
横沥镇 2020 年土地利用分类 - 第五版
改进点：
1. 自适应过采样（所有小样本类别都过采样）
2. 提高样本量 SAMPLES_PER_CLASS = 600
3. 更清晰的类别处理和调试信息
4. 建筑用地自动识别为类别8
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
s2_path = os.path.join(base_dir, r"Sentine2_new_new\S2_Hengli_Fixed_Stack_2020.tif")
clcd_raw = os.path.join(base_dir, r"CLCD\CLCD_v01_2020_albert_guangzhou.tif")
boundary = os.path.join(base_dir, r"hengli_441302110000\hengli_441302110000.shp")

results_dir = os.path.join(base_dir, "Results_Sklearn_13bands_texture_v5")
os.makedirs(results_dir, exist_ok=True)
out_tif = os.path.join(results_dir, "LULC_2020_13bands_texture_v5.tif")
accuracy_file = os.path.join(results_dir, "Accuracy_2020_13bands_texture_v5.txt")

TREES = 300
SAMPLES_PER_CLASS = 600          # 第五版提高样本量
VALIDATION_RATIO = 0.2
BUILTUP_CLASS = 8
MIN_SAMPLES = 50                 # 小于这个数量的类别强制过采样

# 波段名称（必须与影像实际顺序一致）
BAND_NAMES = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B11", "NDVI", "NDWI", "NDBI", "BSI"]
TEXTURE_NAME = "B8_STD"

# ==========================================
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

# 1. 对齐 CLCD
print("[1/6] 对齐 CLCD 到 Sentinel-2 影像...")
s2_r = Raster(s2_path)
arcpy.env.outputCoordinateSystem = s2_r
arcpy.env.snapRaster = s2_path
arcpy.env.extent = s2_r
arcpy.env.cellSize = s2_r.meanCellWidth

clcd_align = ExtractByMask(Raster(clcd_raw), boundary)
clcd_align_path = os.path.join(results_dir, "CLCD_align.tif")
clcd_align.save(clcd_align_path)

# 2. 读取数据 + 计算纹理
print("[2/6] 读取影像数据并计算 B8 纹理...")
clcd_arr = arcpy.RasterToNumPyArray(clcd_align_path, nodata_to_value=-9999)

band_list = []
for i in range(1, 13):   # 前12个波段
    band_arr = arcpy.RasterToNumPyArray(ExtractBand(s2_path, i))
    band_list.append(band_arr)

# 计算 B8 3x3 局部标准差纹理
b8_raw = ExtractBand(s2_path, 4)   # B8 是第4波段
b8_texture = FocalStatistics(b8_raw, NbrRectangle(3, 3, "CELL"), "STD")
texture_arr = arcpy.RasterToNumPyArray(b8_texture)
band_list.append(texture_arr)

BAND_NAMES.append(TEXTURE_NAME)
features = np.stack(band_list, axis=-1).astype(np.float32)
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

rows, cols = clcd_arr.shape
print(f"  影像尺寸: {rows} x {cols}, 特征数: {len(BAND_NAMES)}")

# 3. 分层抽样 + 自适应过采样
print("[3/6] 分层抽样 + 自适应过采样...")
unique_classes = np.unique(clcd_arr[clcd_arr != -9999])
print("实际出现的类别:", sorted(unique_classes))

X_train_list, y_train_list = [], []
X_valid_list, y_valid_list = [], []

for cls in unique_classes:
    r_idx, c_idx = np.where(clcd_arr == cls)
    total = len(r_idx)
    if total < MIN_SAMPLES:
        print(f"  警告：类别 {cls} 样本太少 ({total}个)，跳过或需特殊处理")
        continue

    n_train = int(min(SAMPLES_PER_CLASS, total) * (1 - VALIDATION_RATIO))
    n_valid = min(SAMPLES_PER_CLASS, total) - n_train

    perm = np.random.permutation(total)
    train_idx = perm[:n_train]
    valid_idx = perm[n_train:n_train + n_valid]

    train_feat = features[r_idx[train_idx], c_idx[train_idx]]
    valid_feat = features[r_idx[valid_idx], c_idx[valid_idx]]

    # 自适应过采样：样本少的类别多采样一些
    oversample_factor = 3 if total < 10000 else 2 if total < 50000 else 1
    if cls == BUILTUP_CLASS:
        oversample_factor = max(oversample_factor, 2)

    train_feat = np.repeat(train_feat, oversample_factor, axis=0)
    train_label = np.full(len(train_feat), cls)      # 注意这里用原始 cls，而不是映射后的code

    valid_label = np.full(n_valid, cls)

    X_train_list.append(train_feat)
    y_train_list.append(train_label)
    X_valid_list.append(valid_feat)
    y_valid_list.append(valid_label)

    print(f"  类别 {cls}: 原始 {total} → 训练样本 {len(train_feat)} (过采样×{oversample_factor})")

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)
X_valid = np.vstack(X_valid_list)
y_valid = np.concatenate(y_valid_list)

print(f"\n最终训练样本: {len(y_train)} 个 | 验证样本: {len(y_valid)} 个")

# 4. 训练随机森林
print(f"[4/6] 训练随机森林 ({len(BAND_NAMES)} 个特征, {TREES} 棵树)...")
rf = RandomForestClassifier(n_estimators=TREES, max_depth=30, random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_train, y_train)

# 波段重要性
print("\n波段重要性排序:")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for rank, idx in enumerate(indices, 1):
    print(f"  {rank:2d}. {BAND_NAMES[idx]:8s} : {importances[idx]:.4f}")

# 5. 全图预测（分块）
print("[5/6] 全图预测...")
flat_features = features.reshape(-1, len(BAND_NAMES))
chunk_size = 1024 * 1024
pred_flat = np.zeros(flat_features.shape[0], dtype=np.int16)

for i in range(0, len(pred_flat), chunk_size):
    end = min(i + chunk_size, len(pred_flat))
    pred_flat[i:end] = rf.predict(flat_features[i:end])

pred_arr = pred_flat.reshape(rows, cols)

out_raster = arcpy.NumPyArrayToRaster(pred_arr, arcpy.Point(s2_r.extent.XMin, s2_r.extent.YMax),
                                      s2_r.meanCellWidth, s2_r.meanCellHeight)
out_raster.save(out_tif)
print(f"分类图保存至: {out_tif}")

# 6. 精度评估
print("[6/6] 精度评估...")
y_pred = rf.predict(X_valid)
oa = accuracy_score(y_valid, y_pred)
kappa = cohen_kappa_score(y_valid, y_pred)
report = classification_report(y_valid, y_pred, zero_division=0)

with open(accuracy_file, 'w', encoding='utf-8') as f:
    f.write(f"OA: {oa:.4f}\nKappa: {kappa:.4f}\n\n{report}")

print(f"✅ 第五版完成！ OA = {oa:.4f}, Kappa = {kappa:.4f}")
print(f"精度报告: {accuracy_file}")