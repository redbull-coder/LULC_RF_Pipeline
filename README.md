# 📊 LULC_RF_Pipeline（土地利用/覆盖分类随机森林流程）

> **双语说明 / Bilingual (English / 中文)**

---

## 1️⃣ 项目简介 / Project Overview

**中文：**
本项目实现基于 Sentinel-2 和 CLCD 数据的土地利用/覆盖（LULC）分类流程，使用随机森林分类器，支持全图预测、循环年份处理、精度评估，以及边界裁剪。

**English:**
This project implements a Land Use / Land Cover (LULC) classification pipeline using Sentinel-2 and CLCD data. It leverages a Random Forest classifier, supporting full-scene prediction, multi-year iterative processing, accuracy assessment, and boundary clipping.

---

## 2️⃣ 仓库结构 / Repository Structure

```
scripts/                     # Python 脚本 / Python scripts
├─ .gitkeep                   # 占位文件 / Placeholder
├─ lulc_pipeline_v1.py        # 第1版 / v1 (弃用)
├─ lulc_pipeline_v2.py        # 第2版 / v2
├─ lulc_pipeline_v3.py        # 第3版 / v3
├─ lulc_pipeline_v4.py        # 第4版 / v4
├─ lulc_pipeline_v5.py        # 第5版 / v5


lulc_pipeline_v6.py          # 第6版 / v6 最新版本 / latest version
.gitignore                    # Git 忽略规则 / Git ignore rules
README.md                     # 本 README
gee_lulc_preprocess.js        # GEE 预处理脚本 / GEE preprocessing
```

---

## 3️⃣ 环境依赖 / Dependencies

* **Python ≥3.8**
* **主要库 / Key Libraries:**

```bash
pip install numpy scikit-learn gdal
# ArcGIS Python 环境需要 arcpy
# GEE 脚本在 Google Earth Engine 代码编辑器运行
```

---



## 5️⃣ 版本历史 / Changelog

**中文：**

* **v1 (弃用)**：Error000732，样本进入分类器失败，不符合规范，弃用
* **v2**：放弃 aecpy 预处理，改用 GEE，B2/B3/B4/B8 三波段分类，精度 ~62%，修复类别数字匹配问题，弃用随机撒点
* **v3**：加入 B11、NDVI、NDWI、NDBI、BSI（共 9 波段），加入过采样，精度 ~69%，裸地/建筑/草地仍难区分
* **v4**：修复空值，加入 B5/B6/B7 波段，局部方差代替 GLCM 作为第13波段，精度 ~75%，弃用 NDVI 方差作为第14波段
* **v5**：自适应过采样，RF 树数 300，精度 OA 77%、Kappa 73%，小类别关注增强
* **v6 (当前版本)**：循环处理，全图 GDAL 保存，边界裁剪，直接出图，精度 OA 77%、Kappa 73%

**English:**

* **v1 (Deprecated)**: Error000732, samples failed to enter classifier, did not meet specification
* **v2**: Abandoned aecpy preprocessing, switched to GEE, used B2/B3/B4/B8 3-band classification, accuracy ~62%, fixed class number matching, deprecated random points
* **v3**: Added B11, NDVI, NDWI, NDBI, BSI (9 bands), oversampling included, accuracy ~69%, bare land/building/grass still difficult to separate
* **v4**: Fixed empty bands, added B5/B6/B7, local variance replaced GLCM as 13th band, accuracy ~75%, discarded NDVI variance as 14th band
* **v5**: Adaptive oversampling, RF trees 300, accuracy OA 77%, Kappa 73%, small classes enhanced
* **v6 (current version)**: Iterative multi-year processing, full-scene GDAL saving, boundary clipping, ready for mapping, accuracy OA 77%, Kappa 73%

---

## 6️⃣ Git 忽略规则 / .gitignore

* Python 临时文件：`__pycache__/ *.pyc *.pyo`
* ArcGIS 输出：`*.gdb *.lyr *.lock *.aux *.xml`
* 栅格输出：`*.tif *.tiff *.img`
* GEE 导出文件：`*.csv *.json`
* 本地配置文件：`config_local.yaml`
