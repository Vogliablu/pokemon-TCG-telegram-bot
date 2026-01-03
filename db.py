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
  nickname   TEXT, -- optional
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (user_id, keycode),
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_watchlist_keycode ON watchlist(keycode);

-- Unique nickname per user (only when set)
CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_user_nickname
  ON watchlist(user_id, nickname)
  WHERE nickname IS NOT NULL AND nickname <> '';
"""

async def _migrate(db: aiosqlite.Connection) -> None:
    # Add nickname column if upgrading an existing DB
    cur = await db.execute("PRAGMA table_info(watchlist);")
    cols = [r[1] for r in await cur.fetchall()]
    if "nickname" not in cols:
        await db.execute("ALTER TABLE watchlist ADD COLUMN nickname TEXT;")

    # Ensure the unique index exists (safe to run repeatedly)
    await db.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_user_nickname
          ON watchlist(user_id, nickname)
          WHERE nickname IS NOT NULL AND nickname <> '';
        """
    )
    await db.commit()

async def init_db(db_path: str) -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path, timeout=30)
    await db.execute("PRAGMA foreign_keys = ON;")
    await db.execute("PRAGMA journal_mode = WAL;")
    await db.executescript(SCHEMA)
    await _migrate(db)
    return db
