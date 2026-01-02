import aiosqlite

def normalize_keycode(code: str) -> str:
    return code.strip().upper()

async def ensure_user(db: aiosqlite.Connection, user_id: int):
    await db.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))
    await db.commit()

async def add_watch(db: aiosqlite.Connection, user_id: int, keycode: str):
    keycode = normalize_keycode(keycode)
    await ensure_user(db, user_id)
    await db.execute(
        "INSERT OR IGNORE INTO watchlist(user_id, keycode) VALUES (?, ?)",
        (user_id, keycode),
    )
    await db.commit()

async def remove_watch(db: aiosqlite.Connection, user_id: int, keycode: str):
    keycode = normalize_keycode(keycode)
    await db.execute(
        "DELETE FROM watchlist WHERE user_id = ? AND keycode = ?",
        (user_id, keycode),
    )
    await db.commit()

async def list_watch(db: aiosqlite.Connection, user_id: int) -> list[str]:
    cur = await db.execute(
        "SELECT keycode FROM watchlist WHERE user_id = ? ORDER BY keycode",
        (user_id,),
    )
    rows = await cur.fetchall()
    return [r[0] for r in rows]

async def watchers_for_keycodes(db: aiosqlite.Connection, keycodes: list[str]) -> dict[str, list[int]]:
    # returns {keycode: [user_id, ...]}
    norm = [normalize_keycode(k) for k in keycodes]
    if not norm:
        return {}

    placeholders = ",".join("?" for _ in norm)
    cur = await db.execute(
        f"SELECT keycode, user_id FROM watchlist WHERE keycode IN ({placeholders})",
        tuple(norm),
    )
    rows = await cur.fetchall()

    out: dict[str, list[int]] = {}
    for keycode, user_id in rows:
        out.setdefault(keycode, []).append(user_id)
    return out
