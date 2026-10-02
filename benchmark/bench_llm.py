import json, os, statistics as st, subprocess, sys, time, urllib.request

LLAMA_DIR = os.environ.get('LLAMA_DIR', '/root/llama-b10964/llama-b10964')
MODEL = os.environ.get('LLM_MODEL', '/root/models/gemma-4-E4B-it-Q4_0.gguf')
PROMPT_FILE = os.environ.get('PROMPT_FILE', '/root/bench/prompt.txt')
N_RUNS = int(os.environ.get('N_RUNS', '5'))
N_PREDICT = int(os.environ.get('N_PREDICT', '128'))
THREADS = os.environ.get('THREADS', '8')
PORT = '8090'


def temps():
    out = []
    for i in range(20):
        try:
            out.append(int(open('/sys/class/thermal/thermal_zone%d/temp' % i).read()) / 1000)
        except Exception:
            break
    return max(out) if out else -1


def governor():
    try:
        return open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip()
    except Exception:
        return '?'


def wrap(user):
    if 'gemma' in MODEL.lower():
        return '<|turn>user\n' + user + '<turn|>\n<|turn>model\n'
    return '<|im_start|>user\n' + user + '<|im_end|>\n<|im_start|>assistant\n'


def post(body):
    req = urllib.request.Request('http://127.0.0.1:%s/completion' % PORT, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    user = open(PROMPT_FILE, encoding='utf-8').read().strip()
    env = dict(os.environ, LD_LIBRARY_PATH=LLAMA_DIR)
    log = open('/tmp/bench_srv.log', 'w')
    p = subprocess.Popen([LLAMA_DIR + '/llama-server', '-m', MODEL, '-t', THREADS, '-c', '2048',
                          '--port', PORT, '--no-webui'], stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        t0 = time.time()
        for _ in range(300):
            try:
                urllib.request.urlopen('http://127.0.0.1:%s/health' % PORT, timeout=2)
                break
            except Exception:
                time.sleep(2)
        print('load_s %.1f' % (time.time() - t0), 'governor', governor(), 'temp_start', temps(), flush=True)
        body = {'prompt': wrap(user), 'n_predict': N_PREDICT, 'temperature': 0, 'seed': 1,
                'ignore_eos': True, 'cache_prompt': False}
        rows = []
        for i in range(N_RUNS + 1):
            t = time.time()
            r = post(body)
            wall = time.time() - t
            tm = r['timings']
            row = dict(run=('warmup' if i == 0 else i), wall=wall, prompt_n=tm['prompt_n'],
                       prefill_tps=tm['prompt_per_second'], gen_n=tm['predicted_n'],
                       decode_tps=tm['predicted_per_second'], temp=temps())
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if i > 0:
                rows.append(row)
        if rows:
            print('SAMPLE_OUTPUT', json.dumps(r['content'][:200], ensure_ascii=False))
        for k in ('wall', 'prefill_tps', 'decode_tps'):
            v = [x[k] for x in rows]
            print('SUMMARY %s mean %.2f std %.2f min %.2f max %.2f' % (k, st.mean(v), st.pstdev(v), min(v), max(v)))
        print('SUMMARY prompt_n', rows[0]['prompt_n'], 'gen_n', rows[0]['gen_n'], 'model', os.path.basename(MODEL))
    finally:
        p.terminate()
        p.wait()
    print('ALLDONE')


main()
