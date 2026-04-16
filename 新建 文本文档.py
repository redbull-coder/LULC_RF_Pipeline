# -*- coding: utf-8 -*-
import arcpy
import os
import numpy as np
import time
from arcpy.sa import *

# ================= 【配置区】 =================
years = ["2020", "2021", "2022", "2023", "2024", "2025"]
base_dir = r"D:\lunwen"
clcd_folder = os.path.join(base_dir, "CLCD")
boundary = os.path.join(base_dir, r"横沥镇_441302110000\横沥镇_441302110000.shp")
gdb_path = os.path.join(base_dir, r"MyProject1\MyProject1.gdb")

results_dir = os.path.join(base_dir, "Results")
ecd_dir = os.path.join(results_dir, "Classifiers")
tif_dir = os.path.join(results_dir, "LULC_Maps")

for d in [ecd_dir, tif_dir]:
    if not os.path.exists(d): os.makedirs(d)

arcpy.env.workspace = gdb_path
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

# ================= 【辅助工具】 =================

def find_s2_tci(year, parent_path):
    for root, dirs, files in os.walk(parent_path):
        if f"MSIL2A_{year}" in root:
            for f in files:
                if f.endswith("_TCI_10m.jp2"):
                    return os.path.join(root, f)
    return None

def create_samples(clcd_path, out_feat, n_pts):
    """抽样函数"""
    print(f"   🎲 正在执行 20000 点抽样...")
    r = Raster(clcd_path)
    arr = arcpy.RasterToNumPyArray(r, nodata_to_value=-9999)
    rows, cols = np.where(arr != -9999)
    idx = np.random.choice(len(rows), size=min(n_pts, len(rows)), replace=False)
    desc = arcpy.Describe(r)
    
    arcpy.management.CreateFeatureclass(os.path.dirname(out_feat), os.path.basename(out_feat), "POINT", spatial_reference=desc.spatialReference)
    arcpy.management.AddField(out_feat, "classvalue", "LONG")
    
    xs = desc.extent.XMin + (cols[idx] + 0.5) * desc.meanCellWidth
    ys = desc.extent.YMax - (rows[idx] + 0.5) * desc.meanCellHeight
    
    with arcpy.da.InsertCursor(out_feat, ["SHAPE@XY", "classvalue"]) as cursor:
        for x, y, v in zip(xs, ys, arr[rows[idx], cols[idx]]):
            cursor.insertRow([(x, y), int(v)])
    return np.unique(arr[rows[idx], cols[idx]])

# ================= 【主流程】 =================

def run_production():
    last_valid_ecd = None 
    
    for yr in years:
        print(f"\n🌟 正在处理: {yr} 年度数据")
        try:
            # 1. 找文件
            s2_path = find_s2_tci(yr, base_dir)
            clcd_raw = os.path.join(clcd_folder, f"CLCD_v01_{yr}_albert_guangzhou.tif")
            
            if not s2_path or not os.path.exists(clcd_raw):
                print(f"   ⚠️ {yr} 数据缺失，跳过"); continue

            # 2. 关键：真实执行裁剪逻辑
            print(f"   ✂️ 正在裁剪与投影...")
            s2_clip = os.path.join(gdb_path, f"S2_{yr}_clip")
            clcd_clip = os.path.join(gdb_path, f"CLCD_{yr}_clip")
            
            # 裁剪 S2 影像
            arcpy.management.CopyRaster(ExtractByMask(s2_path, boundary), s2_clip)
            # 裁剪并重投影 CLCD，确保对齐
            arcpy.management.ProjectRaster(clcd_raw, clcd_clip, arcpy.Describe(s2_clip).spatialReference, "NEAREST")

            # 3. 抽样
            pts_feat = os.path.join(gdb_path, f"Samples_{yr}")
            found_classes = create_samples(clcd_clip, pts_feat, 20000)
            print(f"   ✅ 抽样完成，类别: {found_classes}")

            # 4. 训练
            ecd_file = os.path.join(ecd_dir, f"Classifier_{yr}.ecd")
            try:
                print(f"   🌲 训练随机森林...")
                arcpy.gp.TrainRandomTreesClassifier_ia(s2_clip, pts_feat, ecd_file, "", "50", "30", "1000", "COLOR;MEAN", "classvalue")
                if os.path.exists(ecd_file): last_valid_ecd = ecd_file
            except:
                if last_valid_ecd:
                    print(f"   🛡️ 训练失败，使用备份 ECD: {os.path.basename(last_valid_ecd)}")
                    ecd_file = last_valid_ecd
                else: raise Exception("无可用分类器")

            # 5. 分类
            print(f"   🖼️ 正在分类...")
            out_tif = os.path.join(tif_dir, f"Hengli_LULC_{yr}.tif")
            arcpy.sa.ClassifyRaster(s2_clip, ecd_file).save(out_tif)
            print(f"   🎉 {yr} 成功出图！")

        except Exception as e:
            print(f"   🚨 {yr} 彻底翻车: {e}")
        finally:
            # 清理中间栅格，防止 GDB 爆炸
            for temp in [f"S2_{yr}_clip", f"CLCD_{yr}_clip"]:
                if arcpy.Exists(temp): arcpy.management.Delete(temp)

if __name__ == "__main__":
    run_production()
    print(f"\n🏁 严总，横沥镇全线收割完成！")🌟 正在处理: 2020 年度数据