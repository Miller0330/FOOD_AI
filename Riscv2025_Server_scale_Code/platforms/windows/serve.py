"""
用 waitress 啟動網站（Windows 不能用 gunicorn）。

  python platforms/windows/serve.py

環境變數：HOST（預設 127.0.0.1，要讓區網連線才改 0.0.0.0）、PORT（預設 8000；Windows 的 5000 常被系統服務佔用）、THREADS（預設 4）。
PLATFORM 沒設時固定用 windows。
"""
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(CODE_DIR)            # main.py 用相對路徑讀 templates、static、模型與營養表
sys.path.insert(0, CODE_DIR)
os.environ.setdefault('PLATFORM', 'windows')

from waitress import serve   # noqa: E402

import main                  # noqa: E402

if __name__ == '__main__':
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '8000'))
    print(f"[serve] http://{host}:{port}")
    serve(main.app, host=host, port=port, threads=int(os.environ.get('THREADS', '4')))
