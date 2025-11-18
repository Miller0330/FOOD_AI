//  Library webside
//  Seeed-Studio/Grove_4Digital_Display
//  https://github.com/Seeed-Studio/Grove_4Digital_Display?tab=readme-ov-file
//  HX711
//  https://github.com/bogde/HX711

/*將HX711和TM1637的函式庫載入程式中，讓使用者可以直接呼叫裡面已經幫你寫好的類別（class）和函式（function）*/
#include "HX711.h" // 載入 HX711 放大器庫，用於讀取負載感測器數據
#include "TM1637.h"// 載入 TM1637 四位數顯示器庫

// 負載感測器 (load cell) 的引腳定義
const int LOADCELL_DOUT_PIN = 2; // HX711 資料輸出腳位
const int LOADCELL_SCK_PIN = 3;  // HX711 時鐘腳位
const int RESET_BUTTON_PIN = 4;  // Pin 4為磅秤歸零

/* Grove 4位數字顯示器引腳*/
#define CLK 7  // 定義時鐘引腳
#define DIO 6  // 定義數據引腳
TM1637 tm1637(CLK, DIO); // 建立 TM1637 顯示器物件

HX711 scale; // 建立 HX711 物件

/*將開發版的LED在開機過程拉至低電位，等待所有設定初始化完畢，LED將開啟，用來告知使用者初始化完成*/
void setup() {
  Serial.begin(115200);
  Serial.println("HX711 Demo");// 開機序列提示
  pinMode(LED_BUILTIN, OUTPUT);// 內建 LED 指示燈，用來顯示初始化完成狀態
  digitalWrite(LED_BUILTIN, LOW);// 初始化 LED 關閉

  // 初始化按鍵引腳
  pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);  // 使用內建上拉電阻，按下時輸出低電平

  Serial.println("Initializing the scale");
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  // 呼叫 HX711 函式庫裡的 begin()，並把資料腳 (DOUT) 和時鐘腳 (SCK) 的腳位號碼傳進去，完成 HX711 與 Arduino 之間的通訊設定，才能接著進行後續的校正與讀值動作。
  
  tm1637.init();// 初始化 TM1637 顯示器
  tm1637.set(BRIGHT_TYPICAL);  // 設定亮度

  // 初始化電子秤
  Serial.println("Before setting up the scale:");
  scale.set_scale(400.f); // 400.f 是依照你的負載感測器校正資料得出的常數
  scale.tare();  // 重置皮重

  digitalWrite(LED_BUILTIN, HIGH);// 初始化完成，點亮內建 LED
}

/*這段 loop() 的程式主要在持續讀取電子秤重量、顯示並監控「歸零按鍵」*/
void loop() {
  float weight = scale.get_units(1);// 從 HX711 讀取重量 (單次取樣)，返回值已經除以校正因子，單位為克 (g)

  // 打印當前重量到串口
  //Serial.print("Weight: ");
  Serial.println(weight, 1);  // 顯示1位小數
  //Serial.println(" g");

  // 將重量轉換為整數並顯示在Grove 4位數字顯示器上
  int displayWeight = (int)weight;  // 轉換為整數以顯示
  tm1637.display(0, (displayWeight / 1000) % 10);  // 顯示千位
  tm1637.display(1, (displayWeight / 100) % 10);   // 顯示百位
  tm1637.display(2, (displayWeight / 10) % 10);    // 顯示十位
  tm1637.display(3, displayWeight % 10);           // 顯示個位

  // 檢查按鍵是否被按下
  if (digitalRead(RESET_BUTTON_PIN) == LOW) {
    scale.tare();  // 重置皮重
    Serial.println("Scale tared (reset to zero)");
  }

  //delay(500);  // 更新顯示
}
