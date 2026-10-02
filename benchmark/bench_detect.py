import os, statistics as st, sys, time
sys.path.insert(0, os.environ.get('APP_DIR', '/root/riv-app'))
from platforms.genio.detector100 import UECFood100Detector

IMG = os.environ.get('BENCH_IMG', '/root/bench/food.jpg')
N = int(os.environ.get('N_RUNS', '30'))
d = UECFood100Detector()
for _ in range(3):
    d.detect(IMG)
lat = []
for _ in range(N):
    t = time.time()
    r = d.detect(IMG)
    lat.append((time.time() - t) * 1000)
print('detections', len(r), 'top', r[0]['class_name'] if r else None)
print('SUMMARY detect_ms mean %.1f std %.1f min %.1f max %.1f n %d' % (st.mean(lat), st.pstdev(lat), min(lat), max(lat), N))
print('ALLDONE')
