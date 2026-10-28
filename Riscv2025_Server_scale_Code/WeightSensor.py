import os
import serial
import time
import threading
import json
from datetime import datetime

FTDI_FALLBACK_URL = os.environ.get('SCALE_FTDI_URL', 'ftdi://0x0403:0x6010/2')


class WeightSensor:
    def __init__(self, port='/dev/ttyUSB1', baudrate=115200, timeout=1):
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
        self.use_ftdi = False
        self.is_connected = False
        self.current_weight = 0.0
        self.last_update = None
        self.reading_thread = None
        self.stop_reading = False
        
    def connect(self):
        """建立串口連接"""
        try:
            port = self.port
            if port.startswith('ftdi://') or (not os.path.exists(port) and port == '/dev/ttyUSB1'):
                from pyftdi.serialext import serial_for_url
                from pyftdi.usbtools import UsbTools
                # 清掉 pyftdi 快取的舊裝置清單（拔插後裝置編號會變）
                try:
                    UsbTools.release_all_devices()
                except Exception:
                    pass
                try:
                    UsbTools.flush_cache()
                except Exception:
                    pass
                if not port.startswith('ftdi://'):
                    port = FTDI_FALLBACK_URL
                self.ser = serial_for_url(port, baudrate=self.baudrate, timeout=self.timeout)
                self.use_ftdi = True
            else:
                self.ser = serial.Serial(port=port, baudrate=self.baudrate, timeout=self.timeout)
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
                if self.use_ftdi or self.ser.in_waiting > 0:
                    data = self.ser.readline()
                    if not data:
                        continue
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
        # 未連接（例如啟動時磅秤還沒接上）或連線中斷時，最多每 5 秒自動重連一次，不必重啟服務
        now = time.time()
        if not self.is_connected and now - getattr(self, '_last_reconnect', 0) > 5:
            self._last_reconnect = now
            try:
                if self.ser is not None:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None
            self.connect()
        # 每次調用時執行一次讀取；連續讀不到資料（拔線、裝置失效）就視為斷線，下次自動重連
        if self.read_weight() is None:
            self._fail = getattr(self, '_fail', 0) + 1
            if self._fail >= 2 and self.is_connected:
                print("磅秤連續讀取失敗，視為斷線")
                self.is_connected = False
                self.current_weight = 0.0
                self.last_update = None
                self._last_reconnect = 0
                try:
                    if self.ser is not None:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
        else:
            self._fail = 0
        
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

def initialize_weight_sensor(port='/dev/ttyUSB1', baudrate=115200):
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
