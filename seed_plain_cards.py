"""
Bulk seed script for hybrid architecture.

What it does:
- creates N plain-color card images on disk
- inserts rows into SQLite cards table:
    keycode
    name
    image_path
    avg_r, avg_g, avg_b
- telegram_file_id remains NULL (lazy Telegram cache)

Usage:
    python seed_plain_cards.py --db bot.db --out ./seed_images --n 1000
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import aiosqlite
from PIL import Image

from db import init_db
from storage import normalize_keycode


# -----------------------------
# Image generation
# -----------------------------

def make_plain_color_image(
    path: Path,
    rgb: tuple[int, int, int],
    size: tuple[int, int] = (256, 256),
) -> None:
    img = Image.new("RGB", size, rgb)
    img.save(path, format="PNG")


# -----------------------------
# DB insert
# -----------------------------

async def insert_card(
    db: aiosqlite.Connection,
    *,
    keycode: str,
    name: str,
    image_path: str,
    avg_r: float,
    avg_g: float,
    avg_b: float,
) -> None:
    await db.execute(
        """
        INSERT OR REPLACE INTO cards
        (keycode, name, image_path, image_url, telegram_file_id, avg_r, avg_g, avg_b)
        VALUES (?, ?, ?, NULL, NULL, ?, ?, ?)
        """,
        (keycode, name, image_path, avg_r, avg_g, avg_b),
    )


# -----------------------------
# Main seeding logic
# -----------------------------

async def seed(
    db_path: str,
    out_dir: Path,
    n: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    db = await init_db(db_path)

    try:
        for i in range(n):
            # Deterministic but varied colors
            r = (37 * i) % 256
            g = (91 * i) % 256
            b = (173 * i) % 256

            keycode = normalize_keycode(f"PLAIN_{i:04d}")
            name = f"Plain Color Card {i:04d}"

            img_path = out_dir / f"{keycode}.png"
            make_plain_color_image(img_path, (r, g, b))

            await insert_card(
                db,
                keycode=keycode,
                name=name,
                image_path=str(img_path.resolve()),
                avg_r=float(r),
                avg_g=float(g),
                avg_b=float(b),
            )

            if i % 100 == 0:
                print(f"Seeded {i}/{n} cards...")

        await db.commit()
        print(f"\nDone. Seeded {n} cards.")
        print(f"Images written to: {out_dir.resolve()}")
        print("telegram_file_id left NULL (will be cached lazily).")

    finally:
        await db.close()


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="bot.db", help="Path to SQLite DB")
    ap.add_argument("--out", default="./seed_images", help="Output directory for images")
    ap.add_argument("--n", type=int, default=1000, help="Number of cards to generate")

    args = ap.parse_args()

    asyncio.run(seed(args.db, Path(args.out), args.n))


if __name__ == "__main__":
    main()
