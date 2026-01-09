import aiosqlite

# We keep avg_r/g/b for backwards compatibility with the current dummy RGB
# classifier. Embeddings are added as nullable columns and will become the
# primary feature vector once the bot flow is switched.

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
  nickname   TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (user_id, keycode),
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_watchlist_keycode ON watchlist(keycode);

CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_user_nickname
  ON watchlist(user_id, nickname)
  WHERE nickname IS NOT NULL AND nickname <> '';

/*
Hybrid cards catalog:
- image_path / image_url: authoritative location for the full dataset (36k)
- telegram_file_id: lazily filled cache for fast re-sending on Telegram
- avg_r/g/b: legacy dummy feature vector
- embedding: 512-dim float32 vector stored as raw bytes (typically L2-normalized)
*/
CREATE TABLE IF NOT EXISTS cards (
  keycode           TEXT PRIMARY KEY,
  name              TEXT,
  image_path        TEXT,
  image_url         TEXT,
  telegram_file_id  TEXT,
  avg_r             REAL NOT NULL,
  avg_g             REAL NOT NULL,
  avg_b             REAL NOT NULL,
  embedding         BLOB,
  embedding_dim     INTEGER,
  embedding_norm    INTEGER DEFAULT 0,
  embedding_model   TEXT,
  created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cards_avg_rgb ON cards(avg_r, avg_g, avg_b);
"""


async def _migrate(db: aiosqlite.Connection) -> None:
    # ---- watchlist migrations ----
    cur = await db.execute("PRAGMA table_info(watchlist);")
    cols = [r[1] for r in await cur.fetchall()]
    if "nickname" not in cols:
        await db.execute("ALTER TABLE watchlist ADD COLUMN nickname TEXT;")

    await db.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_user_nickname
          ON watchlist(user_id, nickname)
          WHERE nickname IS NOT NULL AND nickname <> '';
        """
    )

    # ---- cards migrations (from older BLOB schema, if any) ----
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cards';"
    )
    exists = await cur.fetchone()

    if exists:
        cur = await db.execute("PRAGMA table_info(cards);")
        card_cols = [r[1] for r in await cur.fetchall()]

        # If legacy schema had image_blob, rebuild to the hybrid schema.
        if "image_blob" in card_cols:
            await db.executescript(
                """
                ALTER TABLE cards RENAME TO cards_old;

                CREATE TABLE cards (
                  keycode           TEXT PRIMARY KEY,
                  name              TEXT,
                  image_path        TEXT,
                  image_url         TEXT,
                  telegram_file_id  TEXT,
                  avg_r             REAL NOT NULL,
                  avg_g             REAL NOT NULL,
                  avg_b             REAL NOT NULL,
                  embedding         BLOB,
                  embedding_dim     INTEGER,
                  embedding_norm    INTEGER DEFAULT 0,
                  embedding_model   TEXT,
                  created_at        TEXT NOT NULL DEFAULT (datetime('now'))
                );

                INSERT INTO cards(
                  keycode, name, image_path, image_url, telegram_file_id,
                  avg_r, avg_g, avg_b,
                  embedding, embedding_dim, embedding_norm, embedding_model,
                  created_at
                )
                SELECT
                  keycode,
                  name,
                  NULL,
                  image_url,
                  NULL,
                  avg_r,
                  avg_g,
                  avg_b,
                  NULL,
                  NULL,
                  0,
                  NULL,
                  created_at
                FROM cards_old;

                DROP TABLE cards_old;

                CREATE INDEX IF NOT EXISTS idx_cards_avg_rgb ON cards(avg_r, avg_g, avg_b);
                """
            )
        else:
            # Ensure newer columns exist (idempotent).
            if "image_path" not in card_cols:
                await db.execute("ALTER TABLE cards ADD COLUMN image_path TEXT;")
            if "image_url" not in card_cols:
                await db.execute("ALTER TABLE cards ADD COLUMN image_url TEXT;")
            if "telegram_file_id" not in card_cols:
                await db.execute("ALTER TABLE cards ADD COLUMN telegram_file_id TEXT;")

            # Embedding columns (new)
            if "embedding" not in card_cols:
                await db.execute("ALTER TABLE cards ADD COLUMN embedding BLOB;")
            if "embedding_dim" not in card_cols:
                await db.execute("ALTER TABLE cards ADD COLUMN embedding_dim INTEGER;")
            if "embedding_norm" not in card_cols:
                await db.execute("ALTER TABLE cards ADD COLUMN embedding_norm INTEGER DEFAULT 0;")
            if "embedding_model" not in card_cols:
                await db.execute("ALTER TABLE cards ADD COLUMN embedding_model TEXT;")

            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_cards_avg_rgb ON cards(avg_r, avg_g, avg_b);"
            )

    await db.commit()


async def init_db(db_path: str) -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path, timeout=30)
    await db.execute("PRAGMA foreign_keys = ON;")
    await db.execute("PRAGMA journal_mode = WAL;")
    await db.executescript(SCHEMA)
    await _migrate(db)
    return db
