"""下載並解壓 UEC FOOD 100（約945MB，非商業研究授權），然後轉成YOLO格式。"""
import subprocess, sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).parent
ZIP = ROOT / "dataset100.zip"
if not (ROOT / "UECFOOD100").exists():
    if not ZIP.exists():
        print("下載 dataset100.zip ...")
        urllib.request.urlretrieve("http://foodcam.mobi/dataset100.zip", ZIP)
    print("解壓縮 ...")
    zipfile.ZipFile(ZIP).extractall(ROOT)
assert (ROOT / "UECFOOD100" / "category.txt").exists(), "解壓後找不到 UECFOOD100/category.txt，請檢查資料夾結構"
subprocess.check_call([sys.executable, str(ROOT / "prepare_dataset_full.py")])
