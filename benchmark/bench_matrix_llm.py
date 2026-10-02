import json, os, re, select, statistics as st, subprocess, sys, time

CLI = '/data/models/gemma4_e4b/MatriXRT-llm-cli'
ENG = '/data/models/gemma4_e4b/packed_engine_e4b_fixed_strip_llmonly.json'
ENG_DIR = '/data/models/gemma4_e4b'
PROMPT_FILE = os.environ.get('PROMPT_FILE', '/tmp/prompt.txt')
N_RUNS = int(os.environ.get('N_RUNS', '5'))
OUT = os.environ.get('OUT', '/tmp/bench_matrix_runs.json')
READY = b'Ask your question'


def read_until(fd, token, timeout, first_after=None):
    buf = b''
    first_t = None
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.5)
        if not r:
            continue
        d = os.read(fd, 65536)
        if not d:
            break
        buf += d
        if first_after is not None and first_t is None:
            i = buf.rfind(first_after)
            if i != -1 and buf[i + len(first_after):].strip():
                first_t = time.time()
        if token in buf[len(buf) - len(d) - len(token):]:
            break
    return buf, first_t


def main():
    q = ' '.join(open(PROMPT_FILE, encoding='utf-8').read().split())
    p = subprocess.Popen([CLI, ENG, '--do_sample', '0'], cwd=ENG_DIR, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    fd = p.stdout.fileno()
    t0 = time.time()
    read_until(fd, READY, 180)
    print('load_s %.1f' % (time.time() - t0), flush=True)
    runs = []
    for i in range(N_RUNS + 1):
        t = time.time()
        p.stdin.write((q + '\n').encode('utf-8'))
        p.stdin.flush()
        buf, first_t = read_until(fd, READY, 180, first_after=b'[A]:')
        end = time.time()
        txt = buf.decode('utf-8', 'ignore')
        k = txt.rfind('[A]:')
        ans = txt[k + 4:] if k != -1 else ''
        ans = ans.split('Ask your question')[0]
        ans = '\n'.join(l for l in ans.split('\n') if not re.match(r'^\[\s*\d+\.\d+\]', l.strip())).strip()
        row = dict(run=('warmup' if i == 0 else i), wall=end - t, ttft=(first_t - t) if first_t else None, answer=ans)
        print(json.dumps({k2: v for k2, v in row.items() if k2 != 'answer'}, ensure_ascii=False), 'chars', len(ans), flush=True)
        runs.append(row)
    json.dump(runs, open(OUT, 'w'), ensure_ascii=False)
    p.stdin.write(b'exit\n')
    p.stdin.flush()
    try:
        p.wait(timeout=20)
    except Exception:
        p.kill()
    print('ALLDONE')


main()
