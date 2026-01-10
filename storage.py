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

async def set_user_prototype_threshold(
    db: aiosqlite.Connection,
    *,
    owner_user_id: int,
    nickname: str,
    threshold: float,
) -> int:
    cur = await db.execute(
        """
        UPDATE user_prototypes
        SET threshold = ?
        WHERE owner_user_id = ? AND nickname = ?
        """,
        (float(threshold), int(owner_user_id), nickname),
    )
    await db.commit()
    return int(cur.rowcount or 0)

async def upsert_single_pending_prototype(
    db: aiosqlite.Connection,
    *,
    token: str,
    owner_user_id: int,
    image_path: str,
    embedding_blob: bytes,
    embedding_dim: int = 512,
    embedding_norm: int = 1,
    embedding_model: str | None = None,
) -> Optional[str]:
    """
    Ensures only ONE pending prototype exists per user.
    Returns the previous pending image_path (if any) so the caller can delete the old file.
    """
    # One transaction: read old, replace row
    await db.execute("BEGIN IMMEDIATE")

    cur = await db.execute(
        "SELECT image_path FROM pending_prototypes WHERE owner_user_id = ?",
        (int(owner_user_id),),
    )
    row = await cur.fetchone()
    old_path = str(row[0]) if row and row[0] else None

    # Delete any existing pending for the user
    await db.execute(
        "DELETE FROM pending_prototypes WHERE owner_user_id = ?",
        (int(owner_user_id),),
    )

    # Insert new pending
    await db.execute(
        """
        INSERT INTO pending_prototypes(
          token, owner_user_id, image_path,
          embedding, embedding_dim, embedding_norm, embedding_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            token,
            int(owner_user_id),
            image_path,
            embedding_blob,
            int(embedding_dim),
            int(embedding_norm),
            embedding_model,
        ),
    )

    await db.commit()
    return old_path


async def get_user_prototype_by_nickname(
    db: aiosqlite.Connection,
    *,
    owner_user_id: int,
    nickname: str,
) -> Optional[tuple]:
    """
    Returns a single prototype row for (owner_user_id, nickname), or None.
    Expected columns: id, nickname, image_path, telegram_file_id, threshold
    Adjust SELECT columns if your schema differs.
    """
    cur = await db.execute(
        """
        SELECT id, nickname, image_path, telegram_file_id, threshold
        FROM user_prototypes
        WHERE owner_user_id = ? AND nickname = ?
        """,
        (int(owner_user_id), nickname),
    )
    row = await cur.fetchone()
    return row

async def delete_all_user_prototypes_return_paths(
    db: aiosqlite.Connection,
    *,
    owner_user_id: int,
) -> list[str]:
    """
    Deletes all watched prototypes for a user and returns their image_path list
    so the caller can delete files from disk.
    """
    cur = await db.execute(
        "SELECT image_path FROM user_prototypes WHERE owner_user_id = ?",
        (int(owner_user_id),),
    )
    rows = await cur.fetchall()
    paths = [str(r[0]) for r in rows if r and r[0]]

    await db.execute(
        "DELETE FROM user_prototypes WHERE owner_user_id = ?",
        (int(owner_user_id),),
    )
    await db.commit()

    return paths
