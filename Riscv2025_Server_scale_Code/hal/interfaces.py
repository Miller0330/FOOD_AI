"""
硬體抽象層（HAL）的介面定義。

應用層（main.py、food_logic.py、模板）只依賴這裡的介面，不直接 import 任何平台專屬模組。
每個平台（platforms/<name>/）提供一個 create()，回傳 Platform 物件：

  - llm：實作 LLMBackend（warmup / get_advice）
  - features：功能旗標（meal100、async_advice）
  - warmup_at_import：True 表示模組載入時就暖機（給 gunicorn 這類 WSGI 伺服器用）；
                      False 表示維持舊行為，只在 `python3 main.py` 的 __main__ 暖機
  - meal100：可選，第二組（100 類）NPU 偵測器的設定；不支援的平台為 None
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol


class LLMBackend(Protocol):
    def warmup(self) -> Any:
        """啟動或準備 LLM（載入引擎、建立 prompt cache…）。回傳值僅供記錄。"""

    def get_advice(self, detections: Dict[str, int], nutrition: Dict[str, float],
                   recommendations: Dict[str, Any], weight: float,
                   name_map: Optional[Dict[str, str]] = None,
                   category_map: Optional[Dict[str, str]] = None) -> str:
        """回傳一段建議文字。name_map/category_map 只有支援 100 類的平台會用到，其他平台可忽略。"""


@dataclass
class Meal100Config:
    """三餐主食（100 類）偵測器設定，目前只有 Matrix（NPU）提供。"""
    detector_factory: Callable[[], Any]     # 回傳有 detect(image_path) -> [{'class_name','confidence','bbox'}] 的物件
    nutrition_path: str                     # 100 類營養資料 JSON
    img_size: int                           # 偵測器座標所在的模型輸入邊長（bbox 除以它得到比例）
    min_confidence: float = 0.3             # 納入營養計算的信心值門檻


@dataclass
class Platform:
    name: str
    llm: LLMBackend
    features: Dict[str, bool] = field(default_factory=lambda: {'meal100': False, 'async_advice': False})
    warmup_at_import: bool = False
    meal100: Optional[Meal100Config] = None
    # 網頁可選的 AI 建議模型 [{'id','label','note'}]，第一個是預設；None 表示不提供切換（目前只有 Genio）
    llm_models: Optional[list] = None
