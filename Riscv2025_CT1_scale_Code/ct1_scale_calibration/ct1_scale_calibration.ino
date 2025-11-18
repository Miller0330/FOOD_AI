
//本範例為HX711的校正程式

#include "HX711.h"

/*HX711 接線設定*/
const int DT_PIN = 2;  // HX711 的 DATA 腳位連接到 Arduino D2
const int SCK_PIN = 3; // HX711 的 SCK  腳位連接到 Arduino D3
const int sample_weight = 1500;  //基準物品的真實重量(公克)

HX711 scale; // 建立 HX711 物件

/*這段程式在 setup 階段完成了 HX711 的通訊初始化、比例因子估算與去皮動作，並透過序列埠提示使用者接下來要放入哪個基準重量，為後續精準校正做好準備。*/
void setup() {
  Serial.begin(115200); // 啟動序列埠，設為 115200 bps
  scale.begin(DT_PIN, SCK_PIN); // 初始化 HX711，傳入 DATA/SCK 腳位
  scale.set_scale();  // 開始取得比例參數
  scale.tare(); // 去皮，將當前讀值設為零點
  Serial.println("Nothing on it."); // 提示：目前秤上無物品
  Serial.println(scale.get_units(10));//第二行顯示去皮後的量測值（取 10 次平均）。
  Serial.println("Please put sapmple object on it..."); //提示放上基準物品
}

/*這段 loop() 裡的程式主要在做「不斷計算並顯示校正係數」，這個迴圈會持續測量並顯示校正參數，協助你找出最精準的 scale.set_scale(...) 值，完成電子秤的校正程序。*/
void loop() {
  float current_weight = scale.get_units(10); // 取得10次數值的平均
  float scale_factor = (current_weight / sample_weight); // 計算校正比例：實際讀值 ÷ 樣本真實重量
  Serial.print("Scale number:  ");
  Serial.println(scale_factor, 0); // 顯示比例參數，記起來，以便用在正式的程式中使用
