import aiosqlite
from typing import Optional, Iterable, Dict, List, Tuple

# -----------------
# Normalization
# -----------------

def normalize_keycode(code: str) -> str:
    return code.strip().upper()

def normalize_nickname(nickname: str) -> str:
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

CardRow = Tuple[
    str, Optional[str], Optional[str], Optional[str], Optional[str],
    float, float, float,
    Optional[bytes], Optional[int], Optional[int], Optional[str],
]

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
        SELECT
          keycode, name, image_path, image_url, telegram_file_id,
          avg_r, avg_g, avg_b,
          embedding, embedding_dim, embedding_norm, embedding_model
        FROM cards
        WHERE keycode IN ({placeholders})
        """,
        tuple(norm),
    )
    rows = await cur.fetchall()
    out: Dict[str, CardRow] = {}
    for row in rows:
        k = str(row[0])
        out[k] = (
            k, row[1], row[2], row[3], row[4],
            float(row[5]), float(row[6]), float(row[7]),
            row[8],
            (int(row[9]) if row[9] is not None else None),
            (int(row[10]) if row[10] is not None else None),
            (str(row[11]) if row[11] is not None else None),
        )
    return out

async def set_card_telegram_file_id(db: aiosqlite.Connection, keycode: str, telegram_file_id: str) -> None:
    keycode = normalize_keycode(keycode)
    await db.execute(
        "UPDATE cards SET telegram_file_id = ? WHERE keycode = ?",
        (telegram_file_id, keycode),
    )
    await db.commit()

# -----------------
# Embeddings (new)
# -----------------

async def set_card_embedding(
    db: aiosqlite.Connection,
    keycode: str,
    embedding: bytes,
    *,
    dim: int = 512,
    normed: bool = True,
    model: str | None = None,
) -> None:
    keycode = normalize_keycode(keycode)
    await db.execute(
        """
        UPDATE cards
        SET embedding = ?, embedding_dim = ?, embedding_norm = ?, embedding_model = ?
        WHERE keycode = ?
        """,
        (embedding, int(dim), 1 if normed else 0, model, keycode),
    )
    await db.commit()

async def iter_card_embeddings(db: aiosqlite.Connection) -> List[Tuple[str, bytes]]:
    cur = await db.execute("SELECT keycode, embedding FROM cards WHERE embedding IS NOT NULL")
    rows = await cur.fetchall()
    return [(str(k), bytes(emb)) for (k, emb) in rows if emb is not None]


import aiosqlite
from typing import Optional, Tuple

# -----------------
# Pending prototypes
# -----------------

async def create_pending_prototype(
    db: aiosqlite.Connection,
    *,
    token: str,
    owner_user_id: int,
    image_path: str,
    embedding_blob: bytes,
    embedding_dim: int = 512,
    embedding_norm: int = 1,
    embedding_model: str | None = None,
) -> None:
    await db.execute(
        """
        INSERT OR REPLACE INTO pending_prototypes(
          token, owner_user_id, image_path,
          embedding, embedding_dim, embedding_norm, embedding_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            token,
            owner_user_id,
            image_path,
            embedding_blob,
            int(embedding_dim),
            int(embedding_norm),
            embedding_model,
        ),
    )
    await db.commit()


async def get_latest_pending_prototype(
    db: aiosqlite.Connection,
    *,
    owner_user_id: int,
) -> Optional[Tuple[str, str, bytes, int, int, Optional[str]]]:
    """
    Returns (token, image_path, embedding_blob, embedding_dim, embedding_norm, embedding_model)
    """
    cur = await db.execute(
        """
        SELECT token, image_path, embedding, embedding_dim, embedding_norm, embedding_model
        FROM pending_prototypes
        WHERE owner_user_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (int(owner_user_id),),
    )
    row = await cur.fetchone()
    if not row:
        return None
    token, image_path, emb, dim, norm, model = row
    return (str(token), str(image_path) if image_path else "", bytes(emb), int(dim), int(norm), str(model) if model else None)


async def delete_pending_prototype(
    db: aiosqlite.Connection,
    *,
    token: str,
    owner_user_id: int,
) -> int:
    cur = await db.execute(
        "DELETE FROM pending_prototypes WHERE token = ? AND owner_user_id = ?",
        (token, int(owner_user_id)),
    )
    await db.commit()
    return int(cur.rowcount or 0)

# -----------------
# User prototypes (persistent)
# -----------------

async def create_user_prototype_from_pending(
    db: aiosqlite.Connection,
    *,
    owner_user_id: int,
    nickname: str,
    image_path: str,
    embedding_blob: bytes,
    embedding_dim: int = 512,
    embedding_norm: int = 1,
    embedding_model: str | None = None,
) -> int:
    """
    Inserts into user_prototypes. Returns new prototype id.
    Raises sqlite constraint error if nickname already exists for the user.
    """
    cur = await db.execute(
        """
        INSERT INTO user_prototypes(
          owner_user_id, nickname, image_path,
          embedding, embedding_dim, embedding_norm, embedding_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(owner_user_id),
            nickname,
            image_path,
            embedding_blob,
            int(embedding_dim),
            int(embedding_norm),
            embedding_model,
        ),
    )
    await db.commit()
    return int(cur.lastrowid)

from pathlib import Path
from typing import Optional, Tuple

async def delete_user_prototype_by_nickname(
    db: aiosqlite.Connection,
    *,
    owner_user_id: int,
    nickname: str,
) -> Optional[str]:
    """
    Deletes a user_prototypes row by (owner_user_id, nickname).
    Returns image_path if deleted, else None.
    """
    cur = await db.execute(
        "SELECT image_path FROM user_prototypes WHERE owner_user_id = ? AND nickname = ?",
        (int(owner_user_id), nickname),
    )
    row = await cur.fetchone()
    if not row:
        return None

    image_path = str(row[0]) if row[0] else ""

    cur = await db.execute(
        "DELETE FROM user_prototypes WHERE owner_user_id = ? AND nickname = ?",
        (int(owner_user_id), nickname),
    )
    await db.commit()

    if (cur.rowcount or 0) <= 0:
        return None

    return image_path
