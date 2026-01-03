import aiosqlite
from typing import Optional, Iterable

def normalize_keycode(code: str) -> str:
    return code.strip().upper()

def normalize_nickname(nickname: str) -> str:
    # Make nickname matching case-insensitive for /unwatch
    return nickname.strip().lower()

async def ensure_user(db: aiosqlite.Connection, user_id: int) -> None:
    await db.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))
    await db.commit()

async def add_watch(
    db: aiosqlite.Connection,
    user_id: int,
    keycode: str,
    nickname: Optional[str] = None,
) -> None:
    """
    Adds (user_id, keycode) to watchlist.
    If nickname is provided, sets/updates it for that (user_id, keycode).
    Nicknames are unique per user (enforced by idx_watchlist_user_nickname).
    """
    keycode = normalize_keycode(keycode)
    nick = normalize_nickname(nickname) if nickname and nickname.strip() else None

    await ensure_user(db, user_id)

    await db.execute(
        """
        INSERT INTO watchlist(user_id, keycode, nickname)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, keycode) DO UPDATE SET
          nickname = COALESCE(excluded.nickname, watchlist.nickname)
        """,
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
    return cur.rowcount

async def remove_watch_by_nickname(db: aiosqlite.Connection, user_id: int, nickname: str) -> int:
    nick = normalize_nickname(nickname)
    cur = await db.execute(
        "DELETE FROM watchlist WHERE user_id = ? AND nickname = ?",
        (user_id, nick),
    )
    await db.commit()
    return cur.rowcount

async def list_watch(db: aiosqlite.Connection, user_id: int) -> list[tuple[str, Optional[str]]]:
    cur = await db.execute(
        "SELECT keycode, nickname FROM watchlist WHERE user_id = ? ORDER BY keycode",
        (user_id,),
    )
    rows = await cur.fetchall()
    return [(r[0], r[1]) for r in rows]

async def watchers_for_keycodes(db: aiosqlite.Connection, keycodes: Iterable[str]) -> dict[str, list[int]]:
    norm = [normalize_keycode(k) for k in keycodes if k and k.strip()]
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
        out.setdefault(keycode, []).append(int(user_id))
    return out

async def clear_watchlist(db: aiosqlite.Connection, user_id: int) -> int:
    cur = await db.execute("DELETE FROM watchlist WHERE user_id = ?", (user_id,))
    await db.commit()
    # rowcount should be correct for DELETE; keep a safe fallback just in case.
    removed = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
    return removed
