/**

* 横沥镇 2020-2025 多源特征影像批量构建（最终正确版）
* ✔ B11 真正重采样到 10m（reproject）
* ✔ 所有指数统一使用同一网格
* ✔ 包含 NDVI / NDWI / NDBI / BSI
  */

// 1. 研究区
var roi = ee.FeatureCollection("projects/green-gearbox-493511-g1/assets/Henli_Town_Boundary");
Map.centerObject(roi, 13);
Map.addLayer(roi, {color: 'red'}, 'ROI');

// 2. 去云函数
function maskS2clouds(image) {
var scl = image.select('SCL');
var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
return image.updateMask(mask)
.divide(10000)
.copyProperties(image, ["system:time_start"]);
}

// 3. 年份列表
var years = [2020, 2021, 2022, 2023, 2024, 2025];

// 4. 主循环
years.forEach(function(year) {

print('处理年份:', year);

// ---- 影像筛选与合成 ----
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
.filterBounds(roi)
.filterDate(year + '-01-01', year + '-12-31')
.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
.map(maskS2clouds);

var s2Composite = s2.median().clip(roi);

// ---- 定义参考投影（10m）----
var refProj = s2Composite.select('B2').projection();

// ---- 原始波段（统一到10m网格）----
var b2 = s2Composite.select('B2').reproject({crs: refProj, scale: 10});
var b3 = s2Composite.select('B3').reproject({crs: refProj, scale: 10});
var b4 = s2Composite.select('B4').reproject({crs: refProj, scale: 10});
var b8 = s2Composite.select('B8').reproject({crs: refProj, scale: 10});

// ---- B11 真正重采样到10m ----
var b11 = s2Composite.select('B11')
.resample('bicubic')
.reproject({crs: refProj, scale: 10})
.rename('B11');

// ---- 指数计算（全部使用统一网格）----
var ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI');
var ndwi = b3.subtract(b8).divide(b3.add(b8)).rename('NDWI');

var ndbi = b11.subtract(b8).divide(b11.add(b8)).rename('NDBI');

var bsi = ee.Image().expression(
'(SWIR + RED - NIR - BLUE) / (SWIR + RED + NIR + BLUE)', {
'SWIR': b11,
'RED': b4,
'NIR': b8,
'BLUE': b2
}).rename('BSI');

// ---- 合成最终特征栈 ----
var finalStack = ee.Image.cat([
b2, b3, b4, b8,
b11,
ndvi, ndwi, ndbi, bsi
]).float();

print('波段列表 ' + year + ':', finalStack.bandNames());

// ---- 导出 ----
Export.image.toDrive({
image: finalStack,
description: 'S2_Hengli_Stack_' + year,
folder: 'GEE_LULC_Final_Project',
fileNamePrefix: 'S2_Hengli_Stack_' + year,
region: roi.geometry(),
scale: 10,
crs: 'EPSG:32649',
maxPixels: 1e13
});

});

print('✅ 所有任务已生成，请在 Tasks 面板点击 Run');
