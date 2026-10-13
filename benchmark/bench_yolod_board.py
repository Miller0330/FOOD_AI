"""Matrix board: NPU detector latency breakdown + val-set outputs for mAP, using matrixrt_yolod.
Run on the board with riv-foodai stopped. Env: BIN, IMG_DIR, OUT_DIR, SOAK_SEC."""
import glob, json, os, re, statistics as st, subprocess, sys, time

BIN = os.environ.get('BIN', '/data/yolod/matrixrt_yolod.new')
IMG_DIR = os.environ.get('IMG_DIR', '/data/val_eval/img')
OUT_DIR = os.environ.get('OUT_DIR', '/data/val_eval')
SOAK_SEC = int(os.environ.get('SOAK_SEC', '600'))
MODEL = '/data/models/uecfood_s512/network_binary.nb'
LABELS = '/data/models/uecfood_s512/labels.txt'
END = re.compile(r'@@END ([\d.]+) ([\d.]+) read=([\d.]+) pre=([\d.]+) dec=([\d.]+)')


def start(extra):
    t = time.perf_counter()
    p = subprocess.Popen([BIN, MODEL, '-c', LABELS, '--imgsz', '512', '--nc', '100', '--cores', '2', '--conf', '0.05'] + extra,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    while not p.stdout.readline().startswith('@@READY'):
        pass
    return p, (time.perf_counter() - t) * 1000


def req(p, path):
    t = time.perf_counter()
    p.stdin.write(path + '\n')
    p.stdin.flush()
    lines = []
    while True:
        l = p.stdout.readline()
        if l.startswith('@@END'):
            m = END.match(l)
            return lines, [float(x) for x in m.groups()], (time.perf_counter() - t) * 1000
        if l.startswith('@@ERR'):
            return lines, None, (time.perf_counter() - t) * 1000
        lines.append(l)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def summ(v):
    return dict(mean=round(st.mean(v), 2), p50=round(pct(v, .5), 2), p95=round(pct(v, .95), 2), p99=round(pct(v, .99), 2),
                max=round(max(v), 2), sd=round(st.pstdev(v), 2))


def temp():
    try:
        return int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
    except Exception:
        return None


def run_config(tag, extra, files):
    p, boot = start(extra)
    for f in files[:5]:
        req(p, f)
    rows, outs = [], []
    t0 = time.time()
    for f in files:
        lines, tm, wall = req(p, f)
        outs.append('IMG %s\n%s' % (os.path.basename(f), ''.join(l for l in lines if 'class=' in l)))
        if tm:
            rows.append((wall,) + tuple(tm))
    elapsed = time.time() - t0
    p.stdin.write('quit\n')
    p.stdin.flush()
    p.wait(5)
    open('%s/out_%s.txt' % (OUT_DIR, tag), 'w', encoding='utf-8').write(''.join(outs))
    cols = list(zip(*rows))
    res = {'n': len(rows), 'startup_ms': round(boot, 1), 'images_per_s': round(len(rows) / elapsed, 1),
           'wall_roundtrip_ms': summ(cols[0]), 'npu_infer_ms': summ(cols[1]), 'in_process_total_ms': summ(cols[2]),
           'imread_ms': summ(cols[3]), 'preprocess_ms': summ(cols[4]), 'decode_nms_ms': summ(cols[5])}
    print('RESULT', tag, json.dumps(res), flush=True)
    return res


def main():
    files = sorted(glob.glob(IMG_DIR + '/*.jpg'))
    print('images', len(files), flush=True)
    results = {}
    boots = []
    for _ in range(5):
        p, b = start([])
        boots.append(b)
        p.stdin.write('quit\n')
        p.stdin.flush()
        p.wait(5)
    results['cold_start_ms'] = [round(b) for b in boots]
    print('COLD', results['cold_start_ms'], flush=True)
    for tag, extra in (('stretch_argmax', []), ('letterbox_argmax', ['--letterbox']),
                       ('letterbox_multi', ['--letterbox', '--multi']), ('stretch_multi', ['--multi'])):
        results[tag] = run_config(tag, extra, files)
    # soak: repeat the val set on the production configuration, sample temperature and rolling latency
    p, _ = start([])
    t_start = time.time()
    last = t_start
    win, trace = [], []
    i = 0
    while time.time() - t_start < SOAK_SEC:
        _, tm, wall = req(p, files[i % len(files)])
        i += 1
        win.append(wall)
        if time.time() - last >= 30:
            trace.append({'t_s': round(time.time() - t_start), 'temp_c': temp(), 'mean_ms': round(st.mean(win), 2), 'p95_ms': round(pct(win, .95), 2), 'n': len(win)})
            print('SOAK', json.dumps(trace[-1]), flush=True)
            win = []
            last = time.time()
    p.stdin.write('quit\n')
    p.stdin.flush()
    results['soak'] = {'seconds': SOAK_SEC, 'images': i, 'trace': trace}
    json.dump(results, open(OUT_DIR + '/bench_results.json', 'w'), indent=1)
    print('ALLDONE', flush=True)


main()
