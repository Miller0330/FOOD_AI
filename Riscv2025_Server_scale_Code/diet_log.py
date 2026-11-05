"""
每日飲食紀錄（SQLite）。與平台、模型無關；只有 features['diet_log'] 為真的平台（目前是 Windows）才會使用。

- meals：每次辨識結果一筆（時間、模式、縮圖路徑、合計營養）
- meal_items：每筆用餐的食物明細（營養數字在寫入時就算好存下，之後營養表更新不會改動歷史）
- summaries：每日 LLM 總結的快取（當天的紀錄沒變就不重新生成）
目前只有單一使用者（user 欄位預設 'default'），之後要加暱稱或個人資料時不必改資料表。
資料庫位置：環境變數 DIET_LOG_DB，沒設就用本檔旁邊的 data/diet_log.db。
"""
import hashlib
import json
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime

NUTRIENT_KEYS = ['calories', 'protein', 'carbs', 'fiber', 'fat']
DEFAULT_USER = 'default'
DB_PATH = os.environ.get('DIET_LOG_DB') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'diet_log.db')

_NUT_COLS = ', '.join(f'{k} REAL NOT NULL DEFAULT 0' for k in NUTRIENT_KEYS)
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL DEFAULT 'default',
    day TEXT NOT NULL,
    eaten_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    image TEXT,
    weight_g REAL NOT NULL DEFAULT 0,
    {_NUT_COLS}
);
CREATE INDEX IF NOT EXISTS idx_meals_user_day ON meals(user, day);
CREATE TABLE IF NOT EXISTS meal_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meal_id INTEGER NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
    food_key TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    count INTEGER NOT NULL DEFAULT 1,
    weight_g REAL NOT NULL DEFAULT 0,
    {_NUT_COLS}
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    day TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    seconds REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_user_day ON chat_messages(user, day, id);
CREATE TABLE IF NOT EXISTS summaries (
    user TEXT NOT NULL,
    day TEXT NOT NULL,
    signature TEXT NOT NULL,
    text TEXT NOT NULL,
    model TEXT,
    seconds REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user, day)
);
"""

_init_lock = threading.Lock()
_ready = False


def _connect():
    global _ready
    with _init_lock:
        if not _ready:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            with closing(sqlite3.connect(DB_PATH, timeout=10)) as c:
                c.execute('PRAGMA journal_mode=WAL')
                c.executescript(_SCHEMA)
            _ready = True
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    return conn


def _totals(items):
    return {k: sum(it.get(k, 0) for it in items) for k in NUTRIENT_KEYS}


def _insert_items(c, meal_id, items):
    cols = ['meal_id', 'food_key', 'name', 'category', 'count', 'weight_g'] + NUTRIENT_KEYS
    c.executemany(
        f"INSERT INTO meal_items({', '.join(cols)}) VALUES({', '.join('?' * len(cols))})",
        [(meal_id, it['key'], it['name'], it.get('category', ''), it.get('count', 1), it['weight_g'],
          *[it.get(k, 0) for k in NUTRIENT_KEYS]) for it in items])


def add_meal(mode, image, weight_g, items, user=DEFAULT_USER, eaten_at=None):
    """寫入一筆用餐紀錄，回傳 meal id。items：[{key,name,category,count,weight_g,calories,protein,...}]"""
    now = eaten_at or datetime.now()
    t = _totals(items)
    with closing(_connect()) as c:
        with c:
            cur = c.execute(
                f"INSERT INTO meals(user, day, eaten_at, mode, image, weight_g, {', '.join(NUTRIENT_KEYS)}) "
                f"VALUES(?, ?, ?, ?, ?, ?, {', '.join('?' * len(NUTRIENT_KEYS))})",
                (user, now.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d %H:%M:%S'), mode, image, weight_g,
                 *[t[k] for k in NUTRIENT_KEYS]))
            _insert_items(c, cur.lastrowid, items)
            return cur.lastrowid


def replace_items(meal_id, items, user=DEFAULT_USER):
    """使用者在結果頁調整份量或換選食物後，用新的明細取代舊的。回傳該筆紀錄是否存在。"""
    t = _totals(items)
    with closing(_connect()) as c:
        with c:
            cur = c.execute(
                f"UPDATE meals SET {', '.join(k + ' = ?' for k in NUTRIENT_KEYS)} WHERE id = ? AND user = ?",
                (*[t[k] for k in NUTRIENT_KEYS], meal_id, user))
            if cur.rowcount == 0:
                return False
            c.execute('DELETE FROM meal_items WHERE meal_id = ?', (meal_id,))
            _insert_items(c, meal_id, items)
            return True


def delete_meal(meal_id, user=DEFAULT_USER):
    with closing(_connect()) as c:
        with c:
            return c.execute('DELETE FROM meals WHERE id = ? AND user = ?', (meal_id, user)).rowcount > 0


def get_day(day, user=DEFAULT_USER):
    """回傳 {'day', 'meals': [...每筆含 items 與 totals], 'totals', 'categories'}。"""
    with closing(_connect()) as c:
        meals = [dict(r) for r in c.execute(
            'SELECT * FROM meals WHERE user = ? AND day = ? ORDER BY eaten_at, id', (user, day))]
        for m in meals:
            m['time'] = m['eaten_at'][11:16]
            m['items'] = [dict(r) for r in c.execute('SELECT * FROM meal_items WHERE meal_id = ? ORDER BY id', (m['id'],))]
            m['totals'] = {k: m[k] for k in NUTRIENT_KEYS}
    return {
        'day': day,
        'meals': meals,
        'totals': _totals([m['totals'] for m in meals]),
        'categories': sorted({it['category'] for m in meals for it in m['items'] if it['category']}),
    }


def signature(day_data):
    """當天紀錄的指紋：新增、刪除或調整份量後會變，用來判斷每日總結快取是否還有效。"""
    key = [[m['id'], *[round(m['totals'][k], 1) for k in NUTRIENT_KEYS]] for m in day_data['meals']]
    return hashlib.sha1(json.dumps(key).encode('utf-8')).hexdigest()


def add_chat_exchange(day, question, answer, model=None, seconds=None, user=DEFAULT_USER):
    """一輪對話（使用者的問題與營養師的回答）一起寫入，避免失敗時留下只有問題沒有回答的紀錄。"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with closing(_connect()) as c:
        with c:
            c.executemany(
                'INSERT INTO chat_messages(user, day, role, content, model, seconds, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)',
                [(user, day, 'user', question, None, None, now),
                 (user, day, 'assistant', answer, model, seconds, now)])


def get_chat(day, limit=None, user=DEFAULT_USER):
    """某天的對話（舊到新）；limit 只取最後 N 則。"""
    with closing(_connect()) as c:
        rows = [dict(r) for r in c.execute(
            'SELECT role, content, model, seconds, created_at FROM chat_messages WHERE user = ? AND day = ? ORDER BY id',
            (user, day))]
    return rows[-limit:] if limit else rows


def clear_chat(day, user=DEFAULT_USER):
    with closing(_connect()) as c:
        with c:
            return c.execute('DELETE FROM chat_messages WHERE user = ? AND day = ?', (user, day)).rowcount


def get_summary(day, sig, user=DEFAULT_USER):
    """取出快取的每日總結；指紋不符（紀錄有變）就回傳 None。"""
    with closing(_connect()) as c:
        row = c.execute('SELECT * FROM summaries WHERE user = ? AND day = ?', (user, day)).fetchone()
    return dict(row) if row and row['signature'] == sig else None


def save_summary(day, sig, text, model=None, seconds=None, user=DEFAULT_USER):
    with closing(_connect()) as c:
        with c:
            c.execute('INSERT OR REPLACE INTO summaries(user, day, signature, text, model, seconds, created_at) '
                      'VALUES(?, ?, ?, ?, ?, ?, ?)',
                      (user, day, sig, text, model, seconds, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
