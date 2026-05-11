/**
 * 横瀝鎮 2020 土地利用分類影像構建（修復版 - 徹底解決 B11 NaN 問題）
 * 🛠 修復重點：
 * 1. 在 map 階段對每一張影像進行 B11 重採樣，確保合成後波段不丟失。
 * 2. 稍微放寬雲量篩選，增加 2020 年的有效像元數量。
 * 3. 增加紅邊波段（B5, B6, B7），這是提升耕地/草地分類精度的核心。
 */

// 1. 定義研究區
var roi = ee.FeatureCollection("projects/green-gearbox-493511-g1/assets/Henli_Town_Boundary");
Map.centerObject(roi, 13);
Map.addLayer(roi, {color: 'red'}, 'ROI');

// 2. 去雲函數（基於 SCL 掩碼）
function maskS2clouds(image) {
  var scl = image.select('SCL');
  // 排除飽和像素、雲、雲影和雪
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return image.updateMask(mask)
    .divide(10000)
    .copyProperties(image, ["system:time_start"]);
}

// 3. 設置處理年份（你可以根據需要調整）
var year = 2020;

// 4. 影像篩選與預處理
var s2Col = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(roi)
  .filterDate(year + '-01-01', year + '-12-31')
  // 稍微放寬雲量，確保有足夠的像元參與合成
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) 
  .map(maskS2clouds)
  .map(function(img) {
    // 獲取當前影像的 10m 參考投影
    var proj10m = img.select('B2').projection();
    
    // 【核心修復】在合成前就對 20m 波段進行重採樣
    var b5 = img.select('B5').resample('bicubic').reproject({crs: proj10m, scale: 10});
    var b6 = img.select('B6').resample('bicubic').reproject({crs: proj10m, scale: 10});
    var b7 = img.select('B7').resample('bicubic').reproject({crs: proj10m, scale: 10});
    var b11 = img.select('B11').resample('bicubic').reproject({crs: proj10m, scale: 10});
    
    // 只保留需要的 10m 波段並加入修復後的 20m 波段
    return img.select(['B2','B3','B4','B8'])
      .addBands([b5, b6, b7, b11]);
  });

// 5. 中值合成（Median Composite）
// 中值合成能有效去除殘留雲影
var s2Composite = s2Col.median().clip(roi);

// 6. 計算光譜指數（確保全部基於修復後的波段）
var b2 = s2Composite.select('B2');
var b3 = s2Composite.select('B3');
var b4 = s2Composite.select('B4');
var b8 = s2Composite.select('B8');
var b11 = s2Composite.select('B11');

var ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI');
var ndwi = b3.subtract(b8).divide(b3.add(b8)).rename('NDWI');
var ndbi = b11.subtract(b8).divide(b11.add(b8)).rename('NDBI');
var bsi = s2Composite.expression(
  '(SWIR + RED - NIR - BLUE) / (SWIR + RED + NIR + BLUE)', {
    'SWIR': b11, 'RED': b4, 'NIR': b8, 'BLUE': b2
  }).rename('BSI');

// 7. 構建特徵棧（包含紅邊波段，提升 F1 分數）
var finalStack = ee.Image.cat([
  b2, b3, b4, b8,           // 10m 原始
  s2Composite.select('B5'), // 紅邊 1
  s2Composite.select('B6'), // 紅邊 2
  s2Composite.select('B7'), // 紅邊 3
  b11,                      // 修復後的 SWIR
  ndvi, ndwi, ndbi, bsi     // 核心指數
]).float();

// 8. 可視化檢查（重要：點擊 Console 旁邊的 Inspector 點擊地圖檢查 B11 是否有值）
print('最終波段列表:', finalStack.bandNames());
Map.addLayer(b11, {min: 0, max: 0.3}, 'B11 檢查（不應為空值）');
Map.addLayer(finalStack, {bands:['B4','B3','B2'], min:0, max:0.3}, '真彩色影像');

// 9. 導出影像
Export.image.toDrive({
  image: finalStack,
  description: 'S2_Hengli_Fixed_Stack_' + year,
  folder: 'GEE_LULC_Final_Project',
  fileNamePrefix: 'S2_Hengli_Fixed_Stack_' + year,
  region: roi.geometry(),
  scale: 10,
  crs: 'EPSG:32649',
  maxPixels: 1e13
});

print('✅ 修復代碼已準備就緒，請點擊 Tasks 面板運行 Export');
