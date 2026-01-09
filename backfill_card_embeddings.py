#!/usr/bin/env python3
"""
backfill_card_embeddings.py

Compute and store 512-D ResNet18 embeddings for each card in the SQLite `cards` table.

Assumptions:
- `cards.image_path` points to a cropped card image on disk (PNG/JPG/...).
- The DB schema includes:
    embedding BLOB,
    embedding_dim INTEGER,
    embedding_norm INTEGER,
    embedding_model TEXT
  (as in the schema upgrade step).

What it does:
- Loads your encoder checkpoint (encoder/last.pt), builds a ResNet18 with fc=Identity.
- Iterates cards from SQLite (optionally only missing embeddings).
- Computes embeddings in batches, L2-normalizes them, and stores float32 bytes in `cards.embedding`.

Usage:
  uv run python backfill_card_embeddings.py --db bot.db --encoder-pt encoder/last.pt

Optional:
  --only-missing     Compute only where embedding IS NULL or embedding_dim != 512
  --device cuda      Use CUDA if available
  --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


# -------------------------
# Image + model utilities
# -------------------------

def pil_open_rgb(path: Path) -> Image.Image:
    # Match your other scripts: ensure RGB even if transparency exists
    return Image.open(path).convert("RGBA").convert("RGB")


def get_embed_transform() -> transforms.Compose:
    # Matches compute_similarity.py
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_resnet18_encoder_from_lastpt(last_pt_path: Path, device: torch.device) -> nn.Module:
    """
    Build ResNet18 -> Linear(num_classes) (to load state) -> then set fc = Identity()
    """
    ckpt = torch.load(last_pt_path, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])

    model = torchvision.models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


# -------------------------
# Dataset
# -------------------------

@dataclass(frozen=True)
class CardItem:
    keycode: str
    image_path: Path


class CardsDataset(Dataset):
    def __init__(self, items: List[CardItem], tf: transforms.Compose):
        self.items = items
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = pil_open_rgb(item.image_path)
        x = self.tf(img)
        return x, item.keycode


# -------------------------
# DB utilities
# -------------------------

def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def fetch_card_items(
    conn: sqlite3.Connection,
    *,
    only_missing: bool,
) -> List[CardItem]:
    if only_missing:
        q = """
        SELECT keycode, image_path
        FROM cards
        WHERE image_path IS NOT NULL
          AND image_path <> ''
          AND (embedding IS NULL OR embedding_dim IS NULL OR embedding_dim != 512)
        """
    else:
        q = """
        SELECT keycode, image_path
        FROM cards
        WHERE image_path IS NOT NULL
          AND image_path <> ''
        """

    rows = conn.execute(q).fetchall()
    items: List[CardItem] = []
    for keycode, image_path in rows:
        if not image_path:
            continue
        p = Path(str(image_path))
        items.append(CardItem(str(keycode), p))
    return items


def update_embeddings_batch(
    conn: sqlite3.Connection,
    batch_rows: List[Tuple[bytes, int, int, Optional[str], str]],
) -> None:
    """
    batch_rows: [(embedding_blob, dim, norm_flag, model_tag, keycode), ...]
    """
    conn.executemany(
        """
        UPDATE cards
        SET embedding = ?, embedding_dim = ?, embedding_norm = ?, embedding_model = ?
        WHERE keycode = ?
        """,
        batch_rows,
    )


# -------------------------
# Main
# -------------------------

@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=str, help="Path to SQLite database (e.g. bot.db)")
    ap.add_argument("--encoder-pt", type=str, default="encoder/last.pt", help="Path to encoder checkpoint last.pt")
    ap.add_argument("--model-tag", type=str, default=None, help="Optional string stored in cards.embedding_model")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="cpu or cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--commit-every", type=int, default=500, help="Commit every N updated rows")
    ap.add_argument("--only-missing", action="store_true", help="Only compute where embedding is missing/wrong dim")
    args = ap.parse_args()

    db_path = args.db
    encoder_pt = Path(args.encoder_pt)

    if not Path(db_path).exists():
        raise SystemExit(f"DB not found: {db_path}")
    if not encoder_pt.exists():
        raise SystemExit(f"Encoder checkpoint not found: {encoder_pt}")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("WARNING: --device cuda requested but CUDA not available; falling back to CPU.")

    conn = connect_db(db_path)

    items = fetch_card_items(conn, only_missing=args.only_missing)
    if not items:
        print("No cards to process (check image_path and/or --only-missing condition).")
        return

    # Filter out missing files; keep a count
    present: List[CardItem] = []
    missing_count = 0
    for it in items:
        if it.image_path.exists():
            present.append(it)
        else:
            missing_count += 1
    if missing_count:
        print(f"WARNING: {missing_count} cards have image_path missing on disk; they will be skipped.")

    if not present:
        print("All candidate image_path files are missing; nothing to do.")
        return

    print(f"Cards to embed: {len(present)} (device={device.type}, batch_size={args.batch_size})")

    encoder = build_resnet18_encoder_from_lastpt(encoder_pt, device)
    tf = get_embed_transform()
    ds = CardsDataset(present, tf)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Process and write
    to_write: List[Tuple[bytes, int, int, Optional[str], str]] = []
    updated = 0

    for x, keycodes in tqdm(loader, desc="Embedding cards"):
        x = x.to(device)
        v = encoder(x).detach().cpu().numpy().astype(np.float32)  # (B, 512)
        # L2 normalize so cosine similarity = dot product
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

        for i, keycode in enumerate(list(keycodes)):
            emb_blob = v[i].tobytes(order="C")  # float32[512]
            to_write.append((emb_blob, 512, 1, args.model_tag, str(keycode)))

        # Flush periodically
        if len(to_write) >= args.commit_every:
            update_embeddings_batch(conn, to_write)
            conn.commit()
            updated += len(to_write)
            to_write.clear()

    if to_write:
        update_embeddings_batch(conn, to_write)
        conn.commit()
        updated += len(to_write)
        to_write.clear()

    conn.close()
    print(f"Done. Updated embeddings for {updated} cards.")


if __name__ == "__main__":
    # Avoid PIL EXIF warnings spamming logs in some environments
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
