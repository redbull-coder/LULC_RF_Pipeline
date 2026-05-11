# -*- coding: utf-8 -*-
"""
横沥镇 2020 年土地利用分类（论文精度优化版）
- 特征组合：红=NDBI，绿=B8，蓝=NDVI（针对建筑/耕地混淆优化）
- 对建筑用地样本进行 2 倍过采样
- 分类后众数滤波平滑
- 分层抽样 + 标签对齐 + 精度评估
"""

import arcpy
import os
import numpy as np
from arcpy.sa import *

# ==========================================
# 配置区
# ==========================================
base_dir = r"D:\lunwen"
s2_path = os.path.join(base_dir, r"Sentine2\Henli_Sentinel_2020.tif")
clcd_raw = os.path.join(base_dir, r"CLCD\CLCD_v01_2020_albert_guangzhou.tif")
boundary = os.path.join(base_dir, r"hengli_441302110000\hengli_441302110000.shp")

gdb_folder = os.path.join(base_dir, "MyProject1")
gdb_path = os.path.join(gdb_folder, "MyProject1.gdb")
results_dir = os.path.join(base_dir, "Results")
ecd_dir = os.path.join(results_dir, "Classifiers")
tif_dir = os.path.join(results_dir, "LULC_Maps")
accuracy_file = os.path.join(results_dir, "Accuracy_2020_Optimized.txt")

TREES = 250                     # 增加树数量
DEPTH = 30
SAMPLES_PER_CLASS = 400
VALIDATION_RATIO = 0.2

# 目标增强的原始 CLCD 类别号（建筑用地为 8）
BUILTUP_CLASS = 8
BUILTUP_OVERSAMPLE_FACTOR = 2   # 样本复制倍数

# ==========================================
# 环境初始化
# ==========================================
for d in [ecd_dir, tif_dir, gdb_folder]:
    if not os.path.exists(d): os.makedirs(d)

if not arcpy.Exists(gdb_path):
    arcpy.management.CreateFileGDB(gdb_folder, "MyProject1.gdb")

arcpy.env.workspace = gdb_path
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

# ==========================================
# 1. 构建针对性三波段特征（NDBI + B8 + NDVI）
# ==========================================
def create_targeted_3band_image(in_raster, out_raster):
    print("[1/5] 构建针对性三波段特征影像...")
    b4 = ExtractBand(in_raster, 1)   # 红
    b3 = ExtractBand(in_raster, 2)   # 绿
    b2 = ExtractBand(in_raster, 3)   # 蓝
    b8 = ExtractBand(in_raster, 4)   # 近红外

    # 计算指数
    ndbi = (b4 - b8) / (b4 + b8)     # 建筑/裸地指数
    ndvi = (b8 - b4) / (b8 + b4)     # 植被指数

    # 组合：红=NDBI，绿=B8，蓝=NDVI（强化建筑与植被区分）
    composite = CompositeBand([ndbi, b8, ndvi])
    composite.save(out_raster)
    print(f"    特征影像已保存: {out_raster}")
    print(f"    波段组合: 红=NDBI, 绿=B8, 蓝=NDVI")
    return out_raster

# ==========================================
# 2. 分层抽样 + 标签重映射 + 建筑用地过采样
# ==========================================
def create_oversampled_stratified_samples(clcd_path, out_train, out_valid, pts_per_class=400, valid_ratio=0.2):
    print("[2/5] 分层抽样、标签重映射与建筑用地过采样...")
    r = Raster(clcd_path)
    cw, ch = r.meanCellWidth, r.meanCellHeight
    arr = arcpy.RasterToNumPyArray(r, nodata_to_value=-9999)
    unique_original = np.unique(arr[arr != -9999])
    unique_original.sort()

    orig_to_code = {orig: i for i, orig in enumerate(unique_original)}
    code_to_orig = {i: orig for orig, i in orig_to_code.items()}
    print(f"    类别映射: {orig_to_code}")

    desc = arcpy.Describe(r)
    train_xs, train_ys, train_vals = [], [], []
    valid_xs, valid_ys, valid_vals = [], [], []

    for orig_cls in unique_original:
        rows, cols = np.where(arr == orig_cls)
        total = len(rows)
        n_train = int(min(pts_per_class, total) * (1 - valid_ratio))
        n_valid = min(pts_per_class, total) - n_train
        if n_train == 0:
            continue

        perm = np.random.permutation(total)
        train_idx = perm[:n_train]
        valid_idx = perm[n_train:n_train+n_valid]

        xs = desc.extent.XMin + (cols + 0.5) * cw
        ys = desc.extent.YMax - (rows + 0.5) * ch

        code_val = orig_to_code[orig_cls]

        # 如果是建筑用地，复制样本以增加权重
        if orig_cls == BUILTUP_CLASS:
            repeat = BUILTUP_OVERSAMPLE_FACTOR
            train_xs.extend(np.repeat(xs[train_idx], repeat))
            train_ys.extend(np.repeat(ys[train_idx], repeat))
            train_vals.extend([code_val] * n_train * repeat)
        else:
            train_xs.extend(xs[train_idx])
            train_ys.extend(ys[train_idx])
            train_vals.extend([code_val] * n_train)

        valid_xs.extend(xs[valid_idx])
        valid_ys.extend(ys[valid_idx])
        valid_vals.extend([code_val] * n_valid)

    _create_point_fc(out_train, train_xs, train_ys, train_vals, desc.spatialReference)
    _create_point_fc(out_valid, valid_xs, valid_ys, valid_vals, desc.spatialReference)

    print(f"    训练样本（过采样后）: {len(train_vals)} 个, 验证样本: {len(valid_vals)} 个")
    return orig_to_code, code_to_orig

def _create_point_fc(out_path, xs, ys, values, sr):
    arcpy.management.CreateFeatureclass(os.path.dirname(out_path), os.path.basename(out_path),
                                        "POINT", spatial_reference=sr)
    arcpy.management.AddField(out_path, "classvalue", "LONG")
    with arcpy.da.InsertCursor(out_path, ["SHAPE@XY", "classvalue"]) as cur:
        for x, y, v in zip(xs, ys, values):
            cur.insertRow([(x, y), v])

# ==========================================
# 3. 提取像元值并清洗
# ==========================================
def extract_values_to_points(pts_fc, raster, out_fc):
    print("[3/5] 提取像元值并过滤无效点...")
    tmp = os.path.join(gdb_path, "tmp_extract")
    ExtractValuesToPoints(pts_fc, raster, tmp)

    valid_points = []
    with arcpy.da.SearchCursor(tmp, ["SHAPE@", "classvalue", "RASTERVALU"]) as cur:
        for row in cur:
            if row[2] is not None and row[2] != 0:
                valid_points.append((row[0], row[1]))

    sr = arcpy.Describe(pts_fc).spatialReference
    arcpy.management.CreateFeatureclass(os.path.dirname(out_fc), os.path.basename(out_fc),
                                        "POINT", spatial_reference=sr)
    arcpy.management.AddField(out_fc, "classvalue", "LONG")
    with arcpy.da.InsertCursor(out_fc, ["SHAPE@", "classvalue"]) as cur:
        for geom, val in valid_points:
            cur.insertRow([geom, val])

    arcpy.management.Delete(tmp)
    return out_fc

# ==========================================
# 4. 训练分类器
# ==========================================
def train_classifier(feature_raster, train_pts, out_ecd, trees=250, depth=30, max_samples=400):
    print("[4/5] 训练随机森林分类器...")
    arcpy.gp.TrainRandomTreesClassifier_ia(
        feature_raster,
        train_pts,
        out_ecd,
        "",
        str(trees),
        str(depth),
        str(max_samples),
        "COLOR;MEAN",
        "classvalue"
    )
    print(f"    分类器已保存: {out_ecd}")
    return out_ecd

# ==========================================
# 5. 分类、平滑并还原标签
# ==========================================
def classify_smooth_remap(classifier_ecd, feature_raster, out_raster_code, out_raster_final, code_to_orig):
    print("[5/5] 执行分类、众数滤波并还原标签...")
    raw_class = "in_memory/raw_class"
    ClassifyRaster(feature_raster, classifier_ecd).save(raw_class)

    # 3x3 众数滤波，减少椒盐噪声
    smoothed = MajorityFilter(raw_class, "FOUR")
    smoothed.save(out_raster_code)

    print("    正在将内部编码还原为原始 CLCD 标签...")
    remap = RemapValue([[code, orig] for code, orig in code_to_orig.items()])
    out_rc = Reclassify(out_raster_code, "Value", remap)
    out_rc.save(out_raster_final)
    return out_raster_final

# ==========================================
# 6. 精度评估（使用平滑后的结果？不对，评估应基于原始预测值）
# ==========================================
def evaluate_accuracy(classifier_ecd, feature_raster, valid_pts_fc, out_report, code_to_orig):
    print("[6/6] 精度评估中...")
    pred_raster = "in_memory/pred_raster"
    ClassifyRaster(feature_raster, classifier_ecd).save(pred_raster)
    tmp_pred = os.path.join(gdb_path, "pred_points")
    ExtractValuesToPoints(valid_pts_fc, pred_raster, tmp_pred)

    y_true, y_pred = [], []
    with arcpy.da.SearchCursor(tmp_pred, ["classvalue", "RASTERVALU"]) as cur:
        for row in cur:
            if row[1] is not None:
                y_true.append(int(row[0]))
                y_pred.append(int(row[1]))

    arcpy.management.Delete(tmp_pred)

    try:
        from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
        cm = confusion_matrix(y_true, y_pred)
        oa = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, labels=sorted(set(y_true)), zero_division=0)

        with open(out_report, 'w', encoding='utf-8') as f:
            f.write(f"总体精度 (OA): {oa:.4f}\n")
            f.write(f"Kappa 系数: {kappa:.4f}\n\n")
            f.write("类别编码映射:\n")
            for code, orig in code_to_orig.items():
                f.write(f"  编码 {code} -> 原始 CLCD 类别 {orig}\n")
            f.write("\n混淆矩阵 (编码值):\n")
            f.write(np.array2string(cm, separator=', '))
            f.write("\n\n分类报告 (编码值):\n")
            f.write(report)
        print(f"    精度报告已保存: {out_report}")
        print(f"    总体精度 = {oa:.4f}, Kappa = {kappa:.4f}")
    except ImportError:
        oa = sum(np.array(y_true) == np.array(y_pred)) / len(y_true) if y_true else 0
        with open(out_report, 'w', encoding='utf-8') as f:
            f.write(f"总体精度 (OA): {oa:.4f}\n")
            f.write("注：未安装 scikit-learn，无法输出完整混淆矩阵。\n")
        print(f"    总体精度 (OA) = {oa:.4f}")

# ==========================================
# 主流程
# ==========================================
print("\n[START] 横沥镇 2020 年土地利用分类（论文精度优化版）")

try:
    # ---- 0. 准备 S2 范围与坐标系 ----
    s2_r = Raster(s2_path)
    s2_sr = arcpy.Describe(s2_r).spatialReference
    s2_extent = arcpy.Describe(s2_r).extent
    cell_size = s2_r.meanCellWidth

    arcpy.env.outputCoordinateSystem = s2_sr
    arcpy.env.snapRaster = s2_path
    arcpy.env.extent = s2_extent
    arcpy.env.cellSize = cell_size

    # ---- 1. 裁剪 CLCD 并对齐 ----
    print("[0/6] 裁剪并重采样 CLCD...")
    clcd_obj = Raster(clcd_raw)
    clcd_clip = ExtractByMask(clcd_obj, boundary)
    temp_tiff = os.path.join(base_dir, "temp_clcd_clip.tif")
    clcd_clip.save(temp_tiff)
    clcd_align = os.path.join(gdb_path, "CLCD_2020_Aligned")
    arcpy.management.CopyRaster(temp_tiff, clcd_align, format="GRID")
    arcpy.management.Delete(temp_tiff)
    print("    CLCD 对齐完成。")

    # ---- 2. 构建针对性特征影像 ----
    feature_raster = os.path.join(gdb_path, "Feature_Targeted_3bands")
    create_targeted_3band_image(s2_path, feature_raster)

    # ---- 3. 抽样（含建筑用地过采样） + 标签映射 ----
    train_pts = os.path.join(gdb_path, "Pts_Train")
    valid_pts = os.path.join(gdb_path, "Pts_Valid")
    orig_to_code, code_to_orig = create_oversampled_stratified_samples(
        clcd_align, train_pts, valid_pts,
        pts_per_class=SAMPLES_PER_CLASS, valid_ratio=VALIDATION_RATIO
    )

    # ---- 4. 提取值并清洗 ----
    train_clean = os.path.join(gdb_path, "Pts_Train_Clean")
    extract_values_to_points(train_pts, feature_raster, train_clean)

    # 统计样本分布
    class_dist = {}
    with arcpy.da.SearchCursor(train_clean, ["classvalue"]) as cur:
        for row in cur:
            class_dist[row[0]] = class_dist.get(row[0], 0) + 1
    print(f"    清洗后训练样本分布（编码值）: {class_dist}")

    # ---- 5. 训练分类器 ----
    ecd_file = os.path.join(ecd_dir, "RF_2020_Optimized.ecd")
    train_classifier(
        feature_raster=feature_raster,
        train_pts=train_clean,
        out_ecd=ecd_file,
        trees=TREES,
        depth=DEPTH,
        max_samples=SAMPLES_PER_CLASS
    )

    # ---- 6. 分类、滤波与还原 ----
    out_tif_code = os.path.join(tif_dir, "Hengli_2020_LULC_Code.tif")
    out_tif_final = os.path.join(tif_dir, "Hengli_2020_LULC_Optimized.tif")
    classify_smooth_remap(ecd_file, feature_raster, out_tif_code, out_tif_final, code_to_orig)

    # ---- 7. 精度评估 ----
    evaluate_accuracy(ecd_file, feature_raster, valid_pts, accuracy_file, code_to_orig)

    print("\n✅ 2020 年优化版分类完成！")
    print(f"分类图（原始标签）: {out_tif_final}")

except Exception as e:
    print(f"\n❌ 报错: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n[清理] 删除临时数据...")
    temp_items = ["CLCD_2020_Aligned", "Feature_Targeted_3bands", "Pts_Train", "Pts_Valid",
                  "Pts_Train_Clean", "tmp_extract", "pred_points"]
    for item in temp_items:
        full = os.path.join(gdb_path, item)
        if arcpy.Exists(full):
            arcpy.management.Delete(full)
    temp_tiff_path = os.path.join(base_dir, "temp_clcd_clip.tif")
    if arcpy.Exists(temp_tiff_path):
        arcpy.management.Delete(temp_tiff_path)
    if 'out_tif_code' in locals() and arcpy.Exists(out_tif_code):
        arcpy.management.Delete(out_tif_code)
    print("脚本结束。")
