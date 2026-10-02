"""
100類UEC-FOOD主食辨識器（Matrix NPU）。

優先使用常駐程序 matrixrt_yolod（模型只載入一次，之後每張約 25 ms）；
程序不存在、啟動失敗或中途出錯時，自動退回舊作法：每次請求以 subprocess 呼叫
matrixrt_yolo（每張約 0.4 s，其中真正的 NPU 推論只有約 12 ms，其餘是載入模型）。

跟原本9類的 Detector.py 是獨立、平行的第二套偵測器（方案C，兩個入口分開）。
"""
import atexit
import os
import re
import select
import subprocess
import threading
import time

MATRIXRT_YOLO_BIN = '/opt/matrix/bin/matrixrt_yolo'
MATRIXRT_YOLOD_BIN = os.environ.get('MATRIXRT_YOLOD_BIN', '/data/yolod/matrixrt_yolod')
MODEL_PATH = '/data/models/uecfood_s512/network_binary.nb'  # YOLO11s/512px；舊版11n/416px在 /data/models/uecfood_full100/
LABELS_PATH = '/data/models/uecfood_s512/labels.txt'
IMG_SIZE = 512
NUM_CLASSES = 100
NPU_CORES = 2
DAEMON_STARTUP_TIMEOUT = 30
DAEMON_REQUEST_TIMEOUT = 15

_DETECTION_RE = re.compile(
    r'#\d+\s+class=(\d+)\s+\(([^)]+)\)\s+score=([\d.]+)\s+bbox=\[([\d.]+),([\d.]+)\s*->\s*([\d.]+),([\d.]+)\]'
)


def _parse(text):
    detections = []
    for match in _DETECTION_RE.finditer(text):
        _, class_name, score, x1, y1, x2, y2 = match.groups()
        detections.append({
            'class_name': class_name,
            'confidence': float(score),
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
        })
    return detections


class _Daemon:
    def __init__(self, conf, iou):
        self.conf, self.iou = conf, iou
        self.proc = None
        self.buf = b''
        self.lock = threading.Lock()

    def _read_until(self, marker, timeout):
        deadline = time.time() + timeout
        while True:
            i = self.buf.find(marker)
            if i != -1:
                j = self.buf.find(b'\n', i)
                if j != -1:
                    text, self.buf = self.buf[:j + 1], self.buf[j + 1:]
                    return text.decode('utf-8', 'replace')
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError('yolod timeout')
            r, _, _ = select.select([self.proc.stdout], [], [], left)
            if not r:
                raise TimeoutError('yolod timeout')
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('yolod exited')
            self.buf += chunk

    def _start(self):
        self.stop()
        self.proc = subprocess.Popen(
            [MATRIXRT_YOLOD_BIN, MODEL_PATH, '-c', LABELS_PATH,
             '--imgsz', str(IMG_SIZE), '--nc', str(NUM_CLASSES), '--cores', str(NPU_CORES),
             '--conf', str(self.conf), '--iou', str(self.iou)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        self.buf = b''
        self._read_until(b'@@READY', DAEMON_STARTUP_TIMEOUT)

    def stop(self):
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            try:
                p.stdin.write(b'quit\n')
                p.stdin.flush()
                p.wait(3)
            except Exception:
                p.kill()


class UECFood100Detector:
    def __init__(self, conf_threshold=0.15, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._daemon = None
        if os.path.exists(MATRIXRT_YOLOD_BIN):
            self._daemon = _Daemon(conf_threshold, iou_threshold)
            atexit.register(self._daemon.stop)

    def detect(self, image_path, output_path=None):
        """回傳 [{'class_name': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}]"""
        if self._daemon is not None:
            try:
                return self._detect_daemon(image_path, output_path)
            except Exception as e:
                print('[detector100] yolod failed (%s), falling back to one-shot matrixrt_yolo' % e, flush=True)
        return self._detect_oneshot(image_path, output_path)

    def _detect_daemon(self, image_path, output_path):
        d = self._daemon
        with d.lock:
            if d.proc is None or d.proc.poll() is not None:
                d._start()
            line = image_path + ('\t' + output_path if output_path else '') + '\n'
            try:
                d.proc.stdin.write(line.encode('utf-8'))
                d.proc.stdin.flush()
                text = ''
                while True:
                    chunk = d._read_until(b'\n', DAEMON_REQUEST_TIMEOUT)
                    if chunk.startswith('@@END'):
                        break
                    if chunk.startswith('@@ERR'):
                        raise RuntimeError(chunk.strip())
                    text += chunk
            except Exception:
                d.stop()
                raise
        return _parse(text)

    def _detect_oneshot(self, image_path, output_path):
        cmd = [
            MATRIXRT_YOLO_BIN, MODEL_PATH, image_path,
            '--imgsz', str(IMG_SIZE),
            '--nc', str(NUM_CLASSES),
            '--cores', str(NPU_CORES),
            '--conf', str(self.conf_threshold),
            '--iou', str(self.iou_threshold),
            '-c', LABELS_PATH,
        ]
        if output_path:
            cmd += ['-o', output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return _parse(result.stdout)


if __name__ == '__main__':
    import sys
    det = UECFood100Detector()
    for r in det.detect(sys.argv[1], output_path='/tmp/detector100_test_out.jpg'):
        print(r)
