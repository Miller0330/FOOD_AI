import serial
import time
import threading
import json
from datetime import datetime

class WeightSensor:
    def __init__(self, port='/dev/ttySIF1', baudrate=115200, timeout=1):
        """
        初始化重量感測器
        
        Args:
            port (str): 串口端口 
            baudrate (int): 波特率
            timeout (int): 讀取超時時間
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_connected = False
        self.current_weight = 0.0
        self.last_update = None
        self.reading_thread = None
        self.stop_reading = False
        
    def connect(self):
        """建立串口連接"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.is_connected = True
            print(f"成功連接到串口: {self.port}")
            return True
        except serial.SerialException as e:
            print(f"串口連接失敗: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"未知錯誤: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """關閉串口連接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.is_connected = False
            print("串口連接已關閉")
    
    def read_weight(self):
        """讀取重量數據（單次讀取，帶等待機制）"""
        if not self.is_connected or not self.ser:
            print("串口未連接")
            return None
        
        try:
            # 清空緩衝區中的舊數據
            self.ser.reset_input_buffer()
            
            # 等待新數據到達（最多等待2秒）
            max_wait_time = 2.0
            start_time = time.time()
            
            while (time.time() - start_time) < max_wait_time:
                if self.ser.in_waiting > 0:
                    data = self.ser.readline()
                    weight_str = data.decode('utf-8', errors='ignore').strip()
                    
                    print(f"收到原始數據: '{weight_str}'")
                    
                    # 解析重量數據
                    try:
                        weight = float(weight_str)
                        self.current_weight = weight
                        self.last_update = datetime.now()
                        print(f"成功解析重量: {weight}g")
                        return weight
                    except ValueError:
                        print(f"無法解析重量數據: {weight_str}")
                        # 繼續等待下一筆數據
                        continue
                
                time.sleep(0.1)  # 短暫休眠避免CPU佔用過高
            
            print("等待超時，未收到有效數據")
            return None
            
        except serial.SerialException as e:
            print(f"讀取串口數據失敗: {e}")
            self.is_connected = False
            return None
        except Exception as e:
            print(f"讀取重量時發生錯誤: {e}")
            return None
    
    def start_continuous_reading(self):
        """開始持續讀取重量數據的線程"""
        if self.reading_thread and self.reading_thread.is_alive():
            return
        
        self.stop_reading = False
        self.reading_thread = threading.Thread(target=self._continuous_read)
        self.reading_thread.daemon = True
        self.reading_thread.start()
        print("開始持續讀取重量數據")
    
    def _continuous_read(self):
        """持續讀取重量數據的內部方法"""
        while not self.stop_reading and self.is_connected:
            weight = self.read_weight()
            if weight is not None:
                print(f"收到數據: {weight}")
            time.sleep(0.1)  # 短暫休眠避免CPU佔用過高
    
    def get_current_weight(self):
        """獲取當前重量（執行一次讀取）"""
        # 每次調用時執行一次讀取
        self.read_weight()
        
        return {
            'weight': self.current_weight,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'is_connected': self.is_connected
        }
    
    def __del__(self):
        """析構函數，確保資源清理"""
        self.disconnect()

# 全局重量感測器實例
weight_sensor = WeightSensor()

def initialize_weight_sensor(port='/dev/ttySIF1', baudrate=115200):
    """初始化重量感測器（不啟動持續讀取）"""
    global weight_sensor
    weight_sensor = WeightSensor(port=port, baudrate=baudrate)
    
    if weight_sensor.connect():
        print("重量感測器連接成功，等待手動讀取")
        return True
    return False

def get_weight_data():
    """獲取重量數據"""
    global weight_sensor
    return weight_sensor.get_current_weight()
