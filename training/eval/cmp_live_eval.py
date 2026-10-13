import json, os, numpy as np
import quant_eval as q
import qe_eval2 as e
names = open('labels_100.txt', encoding='utf-8').read().split()
idx = {n: i for i, n in enumerate(names)}
nut = json.load(open('../hal_wt/Riscv2025_Server_scale_Code/platforms/matrix/nutrition_data_100_uecfood.json', encoding='utf-8'))
cal = {}
for k, v in nut.items():
    en = v.get('english_name', '').replace(' ', '_').replace("'", '').replace('-', '_')
    cal[en] = (v.get('calories', 0), v.get('default_weight_g', 100))
P = {t: json.load(open(f'board_bench/live_{t}.json')) for t in ('stretch', 'letterbox')}
keys = sorted(P['stretch'])
gt_px = e.gt_space('stretch', keys)
gt_cls = {k: {g[0] for g in gt_px[k]} for k in keys}
def conv(p):
    return {k: [(idx[c], s, *b) for c, s, b in v] for k, v in p.items()}
print('--- 框層級（拉伸座標，conf>=0.15）')
for t in P:
    q.summarize(t, conv(P[t]), gt_px, names)
print('\n--- 影像層級（營養門檻 0.3）與營養影響')
print(f"{'':10} {'有偵測':>7} {'召回':>7} {'精確':>7} {'框/圖':>6} {'熱量/圖(預設份量)':>16} {'熱量錯對(圖數)':>10}")
rows = {}
for t in P:
    anyd = rec = hit = tot = 0
    kc = {}
    for k in keys:
        d = [(c, s) for c, s, b in P[t][k] if s >= 0.3]
        tot += len(d); hit += sum(idx[c] in gt_cls[k] for c, s in d)
        anyd += bool(d); rec += any(idx[c] in gt_cls[k] for c, s in d)
        kc[k] = sum(cal.get(c, (0, 100))[0] * cal.get(c, (0, 100))[1] / 100 for c, s in d)
    rows[t] = kc
    n = len(keys)
    print(f"{t:10} {anyd/n:7.1%} {rec/n:7.1%} {hit/max(tot,1):7.1%} {tot/n:6.2f} {np.mean(list(kc.values())):16.0f}")
# 熱量相對「以真實標註類別、預設份量計」的誤差
gtk = {k: sum(cal.get(names[c], (0, 100))[0] * cal.get(names[c], (0, 100))[1] / 100 for c, *_ in gt_px[k]) for k in keys}
for t in P:
    err = [abs(rows[t][k] - gtk[k]) for k in keys]
    rel = [abs(rows[t][k] - gtk[k]) / max(gtk[k], 1) for k in keys]
    print(f"{t:10} 對真實標註的熱量絕對誤差 平均 {np.mean(err):.0f} kcal，中位數 {np.median(err):.0f}，相對誤差中位數 {np.median(rel):.0%}")
a, b = P['stretch'], P['letterbox']
diff = sum(1 for k in keys if {c for c, s, _ in a[k] if s >= .3} != {c for c, s, _ in b[k] if s >= .3})
better = sum(1 for k in keys if abs(rows['letterbox'][k]-gtk[k]) < abs(rows['stretch'][k]-gtk[k]))
worse = sum(1 for k in keys if abs(rows['letterbox'][k]-gtk[k]) > abs(rows['stretch'][k]-gtk[k]))
print(f'\n兩種方法在 0.3 門檻下偵測到的類別集合不同的圖片：{diff}/{len(keys)}；熱量誤差 letterbox 較小 {better} 張、較大 {worse} 張')
