"""
多平台（HAL）整併的測試。不需要任何板子硬體：硬體相關的部分（LLM、NPU 偵測器）用替身取代。
執行方式（在 repo 根目錄）：  python -m unittest discover -s tests -v
需要：flask、opencv-python、numpy、pyserial（與板子上的環境相同的基本套件，不需要 torch 或 opencc）。
"""
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, 'Riscv2025_Server_scale_Code')
TEXT_EXT = ('.py', '.sh', '.txt', '.md')


def _sha(path):
    data = open(path, 'rb').read()
    if path.endswith(TEXT_EXT):                       # Windows 取出時可能被轉成 CRLF，比對前統一
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


class RiscvBaselineProtection(unittest.TestCase):
    """整併不可以改動 RISC-V（原 main）依賴的檔案：內容必須與 main@146e45b 位元組相同。"""

    def test_protected_files_identical_to_main_baseline(self):
        baseline = json.load(open(os.path.join(ROOT, 'tests', 'riscv_baseline_hashes.json'), encoding='utf-8'))['sha256']
        self.assertGreaterEqual(len(baseline), 8)
        for rel, expected in baseline.items():
            with self.subTest(file=rel):
                self.assertEqual(_sha(os.path.join(ROOT, rel)), expected, f'{rel} 與 main 基準不同')

    def test_riscv_llm_paths_untouched(self):
        src = open(os.path.join(APP, 'platforms', 'riscv', 'llm_advisor.py'), encoding='utf-8').read()
        for needle in ("/home/ademo/llama-cli-new", "/home/ademo/qwen2.5-0.5b-instruct-q4_0.gguf",
                       "/home/ademo/sysprompt.cache"):
            self.assertIn(needle, src)


class PlatformSelection(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, APP)
        for m in [m for m in sys.modules if m == 'hal' or m.startswith('hal.')]:
            del sys.modules[m]
        import hal.selector as sel
        self.sel = sel
        self.sel._cache.clear()

    def tearDown(self):
        self.sel._cache.clear()
        sys.path.remove(APP)

    def _name(self, env, exists):
        with mock.patch.dict(os.environ, env, clear=False):
            if 'PLATFORM' not in env:
                os.environ.pop('PLATFORM', None)
            with mock.patch('os.path.exists', side_effect=lambda p: p in exists):
                self.sel._cache.clear()
                return self.sel.selected_platform_name()

    def test_default_is_riscv_when_nothing_is_set(self):
        self.assertEqual(self._name({}, exists=set()), 'riscv')

    def test_env_wins_over_autodetect(self):
        self.assertEqual(self._name({'PLATFORM': 'genio'}, exists={'/opt/matrix/bin/matrixrt_yolo'}), 'genio')

    def test_env_is_case_insensitive(self):
        self.assertEqual(self._name({'PLATFORM': 'MATRIX'}, exists=set()), 'matrix')

    def test_autodetect_each_board(self):
        self.assertEqual(self._name({}, exists={'/opt/matrix/bin/matrixrt_yolo'}), 'matrix')
        self.assertEqual(self._name({}, exists={'/root/llama-b10964/llama-b10964/llama-completion'}), 'genio')
        self.assertEqual(self._name({}, exists={'/home/ademo/llama-cli-new'}), 'riscv')

    def test_unknown_platform_fails_loudly(self):
        with self.assertRaises(ValueError):
            self._name({'PLATFORM': 'jetson'}, exists=set())


class ImportIsolation(unittest.TestCase):
    """選到某平台時，其他平台的程式（含 Matrix 的 NPU 模組）完全不會被載入。"""

    def _run(self, platform, code):
        env = dict(os.environ, PLATFORM=platform, PYTHONIOENCODING='utf-8')   # 程式會印中文，避免 Windows 預設編碼出錯
        r = subprocess.run([sys.executable, '-c', code], cwd=APP, env=env, capture_output=True,
                           text=True, encoding='utf-8')
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_riscv_never_loads_matrix_or_genio(self):
        out = self._run('riscv', "import sys, hal\np = hal.get_platform()\n"
                                 "print(p.name, p.meal100, sorted(m for m in sys.modules if m.startswith('platforms.')))")
        self.assertIn("riscv None", out)
        self.assertNotIn('matrix', out.split('None', 1)[1])
        self.assertNotIn('genio', out.split('None', 1)[1])

    def test_riscv_gemma_is_opt_in(self):
        code = "import hal; p = hal.get_platform(); print(type(p.llm).__name__)"
        self.assertIn('_RiscvLLM', self._run('riscv', code))
        with mock.patch.dict(os.environ, {'RISCV_LLM': 'gemma'}):
            self.assertIn('_GemmaLLM', self._run('riscv', code))


    def test_shim_exposes_legacy_names_for_riscv_and_genio(self):
        for plat in ('riscv', 'genio'):
            with self.subTest(platform=plat):
                out = self._run(plat, "import LLMAdvisor as L\nprint(callable(L.get_llm_advice), callable(L._ensure_prompt_cache))")
                self.assertIn('True True', out)


def _jpeg_bytes():
    import cv2
    import numpy as np
    img = np.full((240, 320, 3), 127, dtype=np.uint8)
    return cv2.imencode('.jpg', img)[1].tobytes()


class _FakeDetector:                       # 9 類偵測器替身（不需要 torch 與模型檔）
    def __init__(self, *a, **k):
        pass

    def detect(self, img):
        return [{'class_name': 'Apple', 'confidence': 0.9, 'bbox': [20, 30, 200, 210]},
                {'class_name': 'Banana', 'confidence': 0.3, 'bbox': [0, 0, 50, 50]}]


class _FakeDetector100:                    # 100 類偵測器替身
    def __init__(self, *a, **k):
        pass

    def detect(self, path):
        return [{'class_name': 'beef_noodle', 'confidence': 0.6, 'bbox': [10.0, 10.0, 400.0, 400.0]}]


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


class AppRoutes(unittest.TestCase):
    """以 Flask test client 端到端驗證三個平台的行為（硬體用替身）。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workdir = os.path.join(self.tmp, 'app')
        shutil.copytree(APP, self.workdir, ignore=shutil.ignore_patterns(
            '*.pt', '*.onnx', '*.tflite', 'llama-cli*', '__pycache__', 'uploads'))
        self.cwd = os.getcwd()
        self.calls = {'warmup': 0, 'advice': []}

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)
        self._purge()
        os.environ.pop('PLATFORM', None)
        while self.workdir in sys.path:
            sys.path.remove(self.workdir)

    @staticmethod
    def _purge():
        for m in [m for m in sys.modules if m in ('main', 'hal', 'food_logic', 'LLMAdvisor', 'Detector', 'Calculate')
                  or m.startswith(('hal.', 'platforms'))]:
            del sys.modules[m]

    def load(self, platform):
        self._purge()
        os.chdir(self.workdir)
        sys.path.insert(0, self.workdir)
        os.environ['PLATFORM'] = platform
        calls = self.calls

        def warm(*a, **k):
            calls['warmup'] += 1
            return True

        def advice(d, n, r, w, name_map=None, category_map=None):
            calls['advice'].append((dict(d), name_map is not None))
            return 'FAKE_ADVICE_TEXT'

        sys.modules['Detector'] = _module('Detector', TFLiteDetector=_FakeDetector)
        llm = _module(f'platforms.{platform}.llm_advisor', get_llm_advice=advice,
                      _ensure_prompt_cache=warm, _ensure_engine_running=warm)
        # 先建立真正的 platforms 套件，再把 llm_advisor 換成替身（避免載入真的 llama-cli / MatriXRT 程式）
        importlib.import_module('platforms')
        importlib.import_module(f'platforms.{platform}') if platform != 'matrix' else None
        if platform == 'matrix':
            sys.modules['platforms.matrix.detector100'] = _module('platforms.matrix.detector100',
                                                                  UECFood100Detector=_FakeDetector100, IMG_SIZE=512)
        if platform == 'riscv' and os.environ.get('RISCV_MEAL100') == '1':
            sys.modules['platforms.riscv.detector100'] = _module(
                'platforms.riscv.detector100', UECFood100Detector=_FakeDetector100, IMG_SIZE=512,
                MODEL_PATH=os.environ['TEST_MODEL'], LABELS_PATH=os.environ['TEST_LABELS'])
        sys.modules[f'platforms.{platform}.llm_advisor'] = llm
        main = importlib.import_module('main')
        return main, main.app.test_client()

    def post(self, client, route='/detect', weight='0'):
        return client.post(route, data={'file': (__import__('io').BytesIO(_jpeg_bytes()), 'a.jpg'), 'weight': weight},
                           content_type='multipart/form-data')

    def test_riscv_matches_matrix_flow(self):
        main, c = self.load('riscv')
        self.assertEqual(main.hw.name, 'riscv')
        self.assertEqual(self.calls['warmup'], 0, 'RISC-V 不可在 import 時暖機（只在 python3 main.py 的 __main__）')
        html = self.post(c).get_data(as_text=True)
        token = re.search(r'data-token="([0-9a-f]+)"', html)
        self.assertIsNotNone(token, '與 Matrix 一致：結果頁先出，建議非同步取得')
        self.assertNotIn('FAKE_ADVICE_TEXT', html)
        j = c.get('/api/advice/' + token.group(1)).get_json()
        self.assertEqual(j['advice'], 'FAKE_ADVICE_TEXT')
        self.assertEqual(c.get('/api/advice/' + token.group(1)).status_code, 404)   # token 只能用一次
        self.assertEqual(c.post('/detect100').status_code, 404)       # 100 類入口不存在
        self.assertNotIn('tab-meal', c.get('/').get_data(as_text=True))
        self.assertEqual(main.MAX_KEPT_UPLOADS, 200)                  # 與 Matrix 一致：保留最近 200 個上傳檔
        self.assertEqual(self.calls['advice'][0][0], {'Apple': 1})    # 信心值 0.3 的 Banana 不列入

    def _meal100_env(self, model_bytes):
        model = os.path.join(self.tmp, 'm.torchscript')
        open(model, 'wb').write(b'0' * model_bytes)
        labels = os.path.join(self.tmp, 'l.txt')
        open(labels, 'w').write('x' + chr(10))
        os.environ.update(RISCV_MEAL100='1', TEST_MODEL=model, TEST_LABELS=labels)

    def test_riscv_meal100_is_opt_in_and_works_when_enabled(self):
        main, c = self.load('riscv')
        self.assertFalse(main.hw.features['meal100'])                 # 預設關閉
        self.assertEqual(c.post('/detect100').status_code, 404)
        self._purge()
        self._meal100_env(1_200_000)
        try:
            main, c = self.load('riscv')
            self.assertTrue(main.hw.features['meal100'])
            self.assertIn('tab-meal', c.get('/').get_data(as_text=True))
            html = self.post(c, '/detect100').get_data(as_text=True)
            self.assertIn('牛肉麵', html)
            self.assertIsNotNone(re.search(r'data-token="([0-9a-f]+)"', html))
        finally:
            for k in ('RISCV_MEAL100', 'TEST_MODEL', 'TEST_LABELS'):
                os.environ.pop(k, None)

    def test_riscv_meal100_stays_off_when_model_is_an_lfs_pointer(self):
        self._meal100_env(130)                                        # Git LFS 指標檔只有約 130 位元組
        try:
            main, c = self.load('riscv')
            self.assertFalse(main.hw.features['meal100'])
            self.assertEqual(c.post('/detect100').status_code, 404)
        finally:
            for k in ('RISCV_MEAL100', 'TEST_MODEL', 'TEST_LABELS'):
                os.environ.pop(k, None)

    def test_riscv_can_fall_back_to_sync_and_no_cleanup(self):
        os.environ['RISCV_ASYNC_ADVICE'] = '0'
        os.environ['MAX_KEPT_UPLOADS'] = '0'
        try:
            main, c = self.load('riscv')
            html = self.post(c).get_data(as_text=True)
            self.assertIn('FAKE_ADVICE_TEXT', html)                   # 退回整併前的同步行為
            self.assertNotIn('data-token', html)
            self.assertEqual(main.MAX_KEPT_UPLOADS, 0)
        finally:
            os.environ.pop('RISCV_ASYNC_ADVICE', None)
            os.environ.pop('MAX_KEPT_UPLOADS', None)

    def test_genio_has_async_advice_and_meal100(self):
        main, c = self.load('genio')
        self.assertEqual(main.hw.name, 'genio')
        self.assertEqual(main.hw.features, {'meal100': True, 'async_advice': True})
        self.assertIsNotNone(re.search(r'data-token="([0-9a-f]+)"', self.post(c).get_data(as_text=True)))
        self.assertEqual(main.MAX_KEPT_UPLOADS, 0)                    # Genio 維持不清理

    def test_matrix_async_and_meal100(self):
        main, c = self.load('matrix')
        self.assertEqual(main.hw.name, 'matrix')
        self.assertEqual(self.calls['warmup'], 1, 'Matrix 由 gunicorn 載入，必須在 import 時暖機')
        self.assertIn('tab-meal', c.get('/').get_data(as_text=True))
        html = self.post(c, '/detect100').get_data(as_text=True)
        token = re.search(r'data-token="([0-9a-f]+)"', html)
        self.assertIsNotNone(token, '應該先出結果、建議非同步取得')
        self.assertIn('牛肉麵', html)
        j = c.get('/api/advice/' + token.group(1)).get_json()
        self.assertTrue(j['success'])
        self.assertEqual(j['advice'], 'FAKE_ADVICE_TEXT')
        self.assertEqual(c.get('/api/advice/' + token.group(1)).status_code, 404)   # token 只能用一次

    def test_calories_consistent_between_tables_and_data_is_usda(self):
        _, c = self.load('riscv')
        html = self.post(c).get_data(as_text=True)
        # 蘋果 1 顆（預設 150g，USDA 57 大卡/100g）≈ 86 大卡；食物明細與營養素表兩邊必須一致
        self.assertGreaterEqual(len(re.findall(r'>86<|\b86 大卡', html)), 2)

    def test_upload_view_image_is_clean_jpg(self):
        _, c = self.load('riscv')
        html = self.post(c).get_data(as_text=True)
        self.assertRegex(html, r'/static/uploads/result_[^"]+\.jpg')
        self.assertIn('det-box', html)                                # 框線由網頁疊圖，不燒進圖片


if __name__ == '__main__':
    unittest.main()
