# -*- coding: utf-8 -*-
"""
横沥镇 2020-2025 年土地利用分类 - 循环版 v6（GDAL保存+边界裁剪）
"""

import arcpy
import os
import numpy as np
from arcpy.sa import *
from osgeo import gdal, osr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

# ==========================================
# 配置区
# ==========================================
base_dir = r"D:\lunwen"
boundary = os.path.join(base_dir, r"hengli_441302110000\hengli_441302110000.shp")

TREES = 300
SAMPLES_PER_CLASS = 600
VALIDATION_RATIO = 0.2
BUILTUP_CLASS = 8
MIN_SAMPLES = 50

BAND_NAMES = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B11", "NDVI", "NDWI", "NDBI", "BSI"]
TEXTURE_NAME = "B8_STD"

label_map = {1:"耕地", 2:"林地", 4:"水域", 5:"草地", 7:"裸地", 8:"建筑用地"}

YEAR_CONFIG = {
    2020: {
        "s2": os.path.join(base_dir, r"Sentine2_new_new\S2_Hengli_Fixed_Stack_2020.tif"),
        "clcd": os.path.join(base_dir, r"CLCD\CLCD_v01_2020_albert_guangzhou.tif"),
    },
    2021: {
        "s2": os.path.join(base_dir, r"Sentine2_new_new\S2_Hengli_Fixed_Stack_2021.tif"),
        "clcd": os.path.join(base_dir, r"CLCD\CLCD_v01_2021_albert_guangzhou.tif"),
    },
    2022: {
        "s2": os.path.join(base_dir, r"Sentine2_new_new\S2_Hengli_Fixed_Stack_2022.tif"),
        "clcd": os.path.join(base_dir, r"CLCD\CLCD_v01_2022_albert_guangzhou.tif"),
    },
    2023: {
        "s2": os.path.join(base_dir, r"Sentine2_new_new\S2_Hengli_Fixed_Stack_2023.tif"),
        "clcd": os.path.join(base_dir, r"CLCD\CLCD_v01_2023_albert_guangzhou.tif"),
    },
    2024: {
        "s2": os.path.join(base_dir, r"Sentine2_new_new\S2_Hengli_Fixed_Stack_2024.tif"),
        "clcd": os.path.join(base_dir, r"CLCD\CLCD_v01_2024_albert_guangzhou.tif"),
    },
    2025: {
        "s2": os.path.join(base_dir, r"Sentine2_new_new\S2_Hengli_Fixed_Stack_2025.tif"),
        "clcd": os.path.join(base_dir, r"CLCD\CLCD_v01_2025_albert_guangzhou.tif"),
    },
}

# ==========================================
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")
np.random.seed(42)

summary_results = []

# ✅ GDAL保存函数
def save_with_gdal(pred_arr, out_tif, s2_r, sr):
    rows, cols = pred_arr.shape
    xmin = s2_r.extent.XMin
    ymax = s2_r.extent.YMax
    cell_w = s2_r.meanCellWidth
    cell_h = s2_r.meanCellHeight

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_tif, cols, rows, 1, gdal.GDT_Int32)
    ds.SetGeoTransform([xmin, cell_w, 0, ymax, 0, -cell_h])

    srs = osr.SpatialReference()
    srs.ImportFromWkt(sr.exportToString())
    ds.SetProjection(srs.ExportToWkt())

    band = ds.GetRasterBand(1)
    band.WriteArray(pred_arr)
    band.SetNoDataValue(0)
    band.FlushCache()
    ds = None

# ✅ 建属性表函数
def build_rat(tif_path):
    arcpy.CalculateStatistics_management(tif_path)
    arcpy.BuildRasterAttributeTable_management(tif_path, "Overwrite")
    arcpy.AddField_management(tif_path, "ClassName", "TEXT", field_length=20)
    with arcpy.da.UpdateCursor(tif_path, ["Value", "ClassName"]) as cursor:
        for row in cursor:
            row[1] = label_map.get(int(row[0]), "未知")
            cursor.updateRow(row)

# ==========================================
# 主循环
# ==========================================
for year, paths in YEAR_CONFIG.items():
    print(f"\n{'='*50}")
    print(f"🚀 开始处理 {year} 年...")
    print(f"{'='*50}")

    s2_path = paths["s2"]
    clcd_raw = paths["clcd"]

    if not os.path.exists(s2_path):
        print(f"⚠️ {year} 年 S2 文件不存在，跳过")
        continue
    if not os.path.exists(clcd_raw):
        print(f"⚠️ {year} 年 CLCD 文件不存在，跳过")
        continue

    results_dir = os.path.join(base_dir, f"Results_v6_{year}")
    os.makedirs(results_dir, exist_ok=True)
    out_tif_raw = os.path.join(results_dir, f"LULC_{year}_raw.tif")
    out_tif_final = os.path.join(results_dir, f"LULC_{year}_final.tif")
    accuracy_file = os.path.join(results_dir, f"Accuracy_{year}_v6.txt")

    try:
        # 1. 对齐CLCD
        print("[1/6] 对齐 CLCD...")
        s2_r = Raster(s2_path)
        arcpy.env.outputCoordinateSystem = s2_r
        arcpy.env.snapRaster = s2_path
        arcpy.env.extent = s2_r
        arcpy.env.cellSize = s2_r.meanCellWidth

        clcd_align = ExtractByMask(Raster(clcd_raw), boundary)
        clcd_align_path = os.path.join(results_dir, f"CLCD_align_{year}.tif")
        clcd_align.save(clcd_align_path)

        # 2. 读取影像+纹理
        print("[2/6] 读取影像...")
        clcd_arr = arcpy.RasterToNumPyArray(clcd_align_path, nodata_to_value=-9999)

        band_list = []
        for i in range(1, 13):
            band_list.append(arcpy.RasterToNumPyArray(ExtractBand(s2_path, i)))

        b8_texture = FocalStatistics(ExtractBand(s2_path, 4), NbrRectangle(3, 3, "CELL"), "STD")
        band_list.append(arcpy.RasterToNumPyArray(b8_texture))

        band_names_year = BAND_NAMES + [TEXTURE_NAME]
        features = np.stack(band_list, axis=-1).astype(np.float32)

        # ✅ 彻底清理极值（这是之前的根本原因）
        features = np.clip(features, -1e6, 1e6)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        rows, cols = clcd_arr.shape
        print(f"  影像尺寸: {rows} x {cols}")

        # 3. 分层抽样
        print("[3/6] 分层抽样...")
        unique_classes = np.unique(clcd_arr[clcd_arr != -9999])
        print("类别:", sorted(unique_classes))

        X_train_list, y_train_list = [], []
        X_valid_list, y_valid_list = [], []

        for cls in unique_classes:
            r_idx, c_idx = np.where(clcd_arr == cls)
            total = len(r_idx)
            if total < MIN_SAMPLES:
                print(f"  类别 {cls} 样本太少({total})，跳过")
                continue

            n_train = int(min(SAMPLES_PER_CLASS, total) * (1 - VALIDATION_RATIO))
            n_valid = min(SAMPLES_PER_CLASS, total) - n_train
            perm = np.random.permutation(total)

            train_feat = features[r_idx[perm[:n_train]], c_idx[perm[:n_train]]]
            valid_feat = features[r_idx[perm[n_train:n_train+n_valid]], c_idx[perm[n_train:n_train+n_valid]]]

            oversample_factor = 3 if total < 10000 else 2 if total < 50000 else 1
            if cls == BUILTUP_CLASS:
                oversample_factor = max(oversample_factor, 2)

            train_feat = np.repeat(train_feat, oversample_factor, axis=0)

            X_train_list.append(train_feat)
            y_train_list.append(np.full(len(train_feat), cls))
            X_valid_list.append(valid_feat)
            y_valid_list.append(np.full(n_valid, cls))

            print(f"  类别 {cls}: {total} → {len(train_feat)} (×{oversample_factor})")

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_valid = np.vstack(X_valid_list)
        y_valid = np.concatenate(y_valid_list)

        # 4. 训练
        print("[4/6] 训练随机森林...")
        rf = RandomForestClassifier(
            n_estimators=TREES, max_depth=30,
            random_state=42, n_jobs=-1, class_weight="balanced"
        )
        rf.fit(X_train, y_train)

        # 5. 预测+保存
        print("[5/6] 全图预测...")
        flat_features = features.reshape(-1, len(band_names_year))
        chunk_size = 512 * 512
        pred_flat = np.zeros(flat_features.shape[0], dtype=np.int32)

        for i in range(0, len(pred_flat), chunk_size):
            end = min(i + chunk_size, len(pred_flat))
            pred_flat[i:end] = np.array(rf.predict(flat_features[i:end]), dtype=np.int32)

        pred_arr = pred_flat.reshape(rows, cols)
        unique_pred = np.unique(pred_arr)
        print(f"  预测唯一值: {unique_pred}")

        if len(unique_pred) <= 1:
            print(f"❌ {year} 预测异常，跳过！")
            continue

        # ✅ GDAL保存原始预测
        sr = arcpy.Describe(s2_path).spatialReference
        save_with_gdal(pred_arr, out_tif_raw, s2_r, sr)
        print(f"  原始预测已保存")

        # ✅ 用边界裁剪，去掉背景
        masked = ExtractByMask(arcpy.Raster(out_tif_raw), boundary)
        masked.save(out_tif_final)
        build_rat(out_tif_final)
        print(f"  裁剪完成: {out_tif_final}")

        # 6. 精度评估
        print("[6/6] 精度评估...")
        y_pred = np.array(rf.predict(X_valid), dtype=np.int32)
        oa = accuracy_score(y_valid, y_pred)
        kappa = cohen_kappa_score(y_valid, y_pred)
        report = classification_report(y_valid, y_pred, zero_division=0)

        with open(accuracy_file, 'w', encoding='utf-8') as f:
            f.write(f"年份: {year}\nOA: {oa:.4f}\nKappa: {kappa:.4f}\n\n{report}")

        summary_results.append((year, oa, kappa))
        print(f"✅ {year} 完成！OA={oa:.4f}, Kappa={kappa:.4f}")

    except Exception as e:
        print(f"❌ {year} 处理失败：{e}")
        import traceback
        traceback.print_exc()
        summary_results.append((year, None, None))
        continue

# 汇总
print(f"\n{'='*50}")
print("📊 精度汇总：")
for year, oa, kappa in summary_results:
    if oa:
        print(f"  {year}: OA={oa:.4f}, Kappa={kappa:.4f}")
    else:
        print(f"  {year}: ❌ 失败")

summary_path = os.path.join(base_dir, "Summary_Accuracy_v7.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    for year, oa, kappa in summary_results:
        f.write(f"{year}: OA={oa:.4f}, Kappa={kappa:.4f}\n" if oa else f"{year}: 失败\n")

print(f"汇总: {summary_path}")
print("🎉 全部完成！")
