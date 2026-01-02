import aiosqlite

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  user_id     INTEGER PRIMARY KEY,
  notify_mode TEXT NOT NULL DEFAULT 'dm',
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS watchlist (
  user_id    INTEGER NOT NULL,
  keycode    TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (user_id, keycode),
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_watchlist_keycode ON watchlist(keycode);
"""

async def init_db(db_path: str) -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path, timeout=30)
    await db.execute("PRAGMA foreign_keys = ON;")
    await db.execute("PRAGMA journal_mode = WAL;")
    await db.executescript(SCHEMA)
    await db.commit()
    return db
