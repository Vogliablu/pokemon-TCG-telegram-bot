import aiosqlite
from typing import Optional, Iterable, Dict, List, Tuple

# -----------------
# Normalization
# -----------------

def normalize_keycode(code: str) -> str:
    return code.strip().upper()

def normalize_nickname(nickname: str) -> str:
    # Case-insensitive nickname matching
    return nickname.strip().lower()

# -----------------
# Users / watchlist
# -----------------

async def ensure_user(db: aiosqlite.Connection, user_id: int) -> None:
    await db.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))
    await db.commit()

async def add_watch(
    db: aiosqlite.Connection,
    user_id: int,
    keycode: str,
    *,
    nickname: Optional[str] = None,
) -> None:
    keycode = normalize_keycode(keycode)
    nick = normalize_nickname(nickname) if nickname else None

    await db.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))
    await db.execute(
        "INSERT OR REPLACE INTO watchlist(user_id, keycode, nickname) VALUES (?, ?, ?)",
        (user_id, keycode, nick),
    )
    await db.commit()

async def remove_watch(db: aiosqlite.Connection, user_id: int, keycode: str) -> int:
    keycode = normalize_keycode(keycode)
    cur = await db.execute(
        "DELETE FROM watchlist WHERE user_id = ? AND keycode = ?",
        (user_id, keycode),
    )
    await db.commit()
    return int(cur.rowcount or 0)

async def remove_watch_by_nickname(db: aiosqlite.Connection, user_id: int, nickname: str) -> int:
    nick = normalize_nickname(nickname)
    cur = await db.execute(
        "DELETE FROM watchlist WHERE user_id = ? AND nickname = ?",
        (user_id, nick),
    )
    await db.commit()
    return int(cur.rowcount or 0)

async def clear_watchlist(db: aiosqlite.Connection, user_id: int) -> int:
    cur = await db.execute("DELETE FROM watchlist WHERE user_id = ?", (user_id,))
    await db.commit()
    return int(cur.rowcount or 0)

# Optional helper used by group flow
async def watchers_for_keycodes(
    db: aiosqlite.Connection, keycodes: Iterable[str]
) -> Dict[str, List[int]]:
    norm = [normalize_keycode(k) for k in keycodes if k and k.strip()]
    if not norm:
        return {}

    placeholders = ",".join("?" for _ in norm)
    cur = await db.execute(
        f"SELECT keycode, user_id FROM watchlist WHERE keycode IN ({placeholders})",
        tuple(norm),
    )
    rows = await cur.fetchall()

    out: Dict[str, List[int]] = {}
    for keycode, user_id in rows:
        out.setdefault(str(keycode), []).append(int(user_id))
    return out

# -----------------
# Hybrid cards catalog
# -----------------

CardRow = Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], float, float, float]
# (keycode, name, image_path, image_url, telegram_file_id, avg_r, avg_g, avg_b)

async def get_card_features(db: aiosqlite.Connection) -> List[Tuple[str, float, float, float]]:
    cur = await db.execute("SELECT keycode, avg_r, avg_g, avg_b FROM cards")
    rows = await cur.fetchall()
    return [(str(k), float(r), float(g), float(b)) for (k, r, g, b) in rows]

async def get_cards_by_keycodes(db: aiosqlite.Connection, keycodes: Iterable[str]) -> Dict[str, CardRow]:
    norm = [normalize_keycode(k) for k in keycodes if k and k.strip()]
    if not norm:
        return {}

    placeholders = ",".join("?" for _ in norm)
    cur = await db.execute(
        f"""
        SELECT keycode, name, image_path, image_url, telegram_file_id, avg_r, avg_g, avg_b
        FROM cards
        WHERE keycode IN ({placeholders})
        """,
        tuple(norm),
    )
    rows = await cur.fetchall()
    out: Dict[str, CardRow] = {}
    for row in rows:
        k = str(row[0])
        out[k] = (k, row[1], row[2], row[3], row[4], float(row[5]), float(row[6]), float(row[7]))
    return out

async def set_card_telegram_file_id(db: aiosqlite.Connection, keycode: str, telegram_file_id: str) -> None:
    keycode = normalize_keycode(keycode)
    await db.execute(
        "UPDATE cards SET telegram_file_id = ? WHERE keycode = ?",
        (telegram_file_id, keycode),
    )
    await db.commit()
