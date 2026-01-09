import os
import io
import tempfile
import logging
from dotenv import load_dotenv
from typing import Final,Iterable, List, Set, Dict, Tuple, Optional
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand,
    InputFile, InputMediaPhoto
)
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.error import TelegramError, Forbidden, BadRequest
import aiosqlite
from db import init_db
import sqlite3
from storage import (
    add_watch as db_add_watch,
    remove_watch as db_remove_watch,
    remove_watch_by_nickname as db_remove_watch_by_nickname,
    clear_watchlist as db_clear_watchlist,
    normalize_keycode,
    get_card_features as db_get_card_features,
    get_cards_by_keycodes as db_get_cards_by_keycodes,
    set_card_telegram_file_id as db_set_card_telegram_file_id,
)

import asyncio
import urllib.request
from pathlib import Path
from PIL import Image, ImageStat
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO

VISION_ROOT = Path(__file__).resolve().parent / "vision"

USER_CROPPER_MODEL_PATH = VISION_ROOT / "user_cropper" / "weights" / "best.pt"
USER_CROPPER_CONF = float(os.getenv("USER_CROPPER_CONF", "0.60"))
USER_CROPPER_MAX_CROPS = int(os.getenv("USER_CROPPER_MAX_CROPS", "3"))
ENCODER_PATH = VISION_ROOT / "encoder" / "last.pt"
USER_CROPPER_PATH = VISION_ROOT / "user_cropper" / "weights" / "best.pt"
TEL_CROPPER_PATH  = VISION_ROOT / "tel_cropper" / "weights" / "best.pt"
EMBED_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

# Silence noisy dependencies that leak or spam
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# (optional but recommended)
logging.getLogger("telegram").setLevel(logging.INFO)


logger = logging.getLogger(__name__)
logger.info("Logging is working")

load_dotenv()
DB_PATH = os.getenv("SQLITE_PATH", "bot.db")
logger.info("Using DB_PATH=%s", DB_PATH)
TOKEN: Final = os.environ["BOT_TOKEN"]
BOT_USERNAME: Final = '@Pokemon_Card_tracker_bot'
CARD_RATIO_PORTRAIT = 63 / 88

# Helpers
def _image_looks_like_single_card(img: Image.Image, tol: float = 0.18) -> bool:
    w, h = img.size
    if w < 20 or h < 20:
        return False
    r = w / h
    rp = CARD_RATIO_PORTRAIT
    rl = 1.0 / rp
    return (abs(r - rp) <= tol) or (abs(r - rl) <= tol)

def _clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))

def _pil_open_rgb_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGBA").convert("RGB")

def _crop_user_cards_obb_to_aabb(
    *,
    user_cropper,
    image_bytes: bytes,
    conf: float,
    max_crops: int,
) -> Tuple[List[bytes], int]:
    """
    Returns (crop_bytes_list, total_detections_found).

    Crop logic = crop_user_pictures.py:
      - detect OBB
      - convert OBB corners -> AABB via min/max
      - clamp to image bounds
      - crop with PIL
    :contentReference[oaicite:4]{index=4}
    """
    img = _pil_open_rgb_from_bytes(image_bytes)
    w_img, h_img = img.size

    # Run YOLO on a temp file (most robust with ultralytics)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        img.save(tmp.name, format="JPEG", quality=95)
        results = user_cropper.predict(source=tmp.name, conf=conf, verbose=False)

    if not results:
        results = []

    if not results or getattr(results[0], "obb", None) is None or len(results[0].obb) == 0:
        # Optional fallback: if photo is basically a scan, treat whole image as one "crop"
        if _image_looks_like_single_card(img):
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            return [bio.getvalue()], 0
        return [], 0

    r0 = results[0]
    total = len(r0.obb)

    corners = r0.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2) :contentReference[oaicite:5]{index=5}

    # Optional: also use conf if present
    confs = None
    if hasattr(r0.obb, "conf") and r0.obb.conf is not None:
        confs = r0.obb.conf.cpu().numpy().astype(np.float32)

    dets = []
    for i, c in enumerate(corners):
        x_min = float(np.min(c[:, 0]))
        x_max = float(np.max(c[:, 0]))
        y_min = float(np.min(c[:, 1]))
        y_max = float(np.max(c[:, 1]))

        x1 = _clamp_int(x_min, 0, w_img - 1)
        y1 = _clamp_int(y_min, 0, h_img - 1)
        x2 = _clamp_int(x_max, x1 + 1, w_img)
        y2 = _clamp_int(y_max, y1 + 1, h_img)

        area = (x2 - x1) * (y2 - y1)
        det_conf = float(confs[i]) if confs is not None else 1.0
        dets.append((det_conf, area, (x1, y1, x2, y2)))

    # Sort by confidence then area (descending)
    dets.sort(key=lambda t: (t[0], t[1]), reverse=True)

    if max_crops > 0:
        dets = dets[:max_crops]

    crop_bytes_list: List[bytes] = []
    for det_conf, area, (x1, y1, x2, y2) in dets:
        crop = img.crop((x1, y1, x2, y2))
        # Basic sanity: skip microscopic crops
        if crop.size[0] < 40 or crop.size[1] < 40:
            continue
        bio = io.BytesIO()
        crop.save(bio, format="PNG")
        crop_bytes_list.append(bio.getvalue())

    return crop_bytes_list, total

def nickname_pending(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    return bool(context.user_data.get("pending_nickname_keycode"))

def normalize_keycode(code: str) -> str:
    return code.strip().upper()

def avg_rgb_from_bytes(image_bytes: bytes) -> tuple[float, float, float]:
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        stat = ImageStat.Stat(im)
        r, g, b = stat.mean
        return float(r), float(g), float(b)

def build_resnet18_encoder_from_lastpt(last_pt: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(last_pt, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])

    model = torchvision.models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


async def _read_file_bytes(path: str) -> bytes:
    p = Path(path)
    return await asyncio.to_thread(p.read_bytes)

async def _download_url_bytes(url: str) -> bytes:
    def _dl() -> bytes:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; cardbot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            return resp.read()
    return await asyncio.to_thread(_dl)

async def load_card_image_bytes(image_path: str | None, image_url: str | None) -> bytes:
    if image_path and image_path.strip():
        return await _read_file_bytes(image_path.strip())
    if image_url and image_url.strip():
        return await _download_url_bytes(image_url.strip())
    raise ValueError("Card has neither image_path nor image_url")

async def ensure_card_index(context: ContextTypes.DEFAULT_TYPE, db: aiosqlite.Connection) -> list[tuple[str, float, float, float]]:
    """
    Cache card features in memory for faster classification.
    context.application.bot_data["card_index"] = [(keycode, r,g,b), ...]
    """
    idx = context.application.bot_data.get("card_index")
    if isinstance(idx, list) and idx:
        return idx

    rows = await db_get_card_features(db)
    context.application.bot_data["card_index"] = rows
    return rows

async def classify_image(
    db: aiosqlite.Connection,
    context: ContextTypes.DEFAULT_TYPE,
    image_bytes: bytes,
    *,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Embedding-based image classifier.

    Returns:
        List of (keycode, similarity), sorted by descending similarity.
    """

    import io
    import numpy as np
    import torch
    from PIL import Image

    encoder = context.application.bot_data["encoder"]
    device = context.application.bot_data["device"]
    emb_matrix = context.application.bot_data["emb_matrix"]
    emb_keycodes = context.application.bot_data["emb_keycodes"]

    # --- bytes → tensor ---
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        x = EMBED_TF(im).unsqueeze(0).to(device)

    # --- embedding ---
    with torch.no_grad():
        v = encoder(x).detach().cpu().numpy().astype(np.float32)[0]

    # L2 normalize
    v /= (np.linalg.norm(v) + 1e-12)

    # --- cosine similarity ---
    sims = emb_matrix @ v  # shape (N,)

    # --- top-k ---
    if top_k >= sims.shape[0]:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]

    return [(str(emb_keycodes[i]), float(sims[i])) for i in idx]


async def watchers_for_keycodes(db: aiosqlite.Connection, keycodes: Iterable[str]) -> Dict[str, List[int]]:
    """
    Returns mapping: {keycode: [user_id, ...]} for watchers.
    """
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
        out.setdefault(keycode, []).append(int(user_id))
    return out

async def add_watch(db: aiosqlite.Connection, user_id: int, keycode: str) -> None:
    keycode = normalize_keycode(keycode)
    await db.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))
    await db.execute(
        "INSERT OR IGNORE INTO watchlist(user_id, keycode) VALUES (?, ?)",
        (user_id, keycode),
    )
    await db.commit()

def build_watch_keyboard(keycodes: List[str], max_buttons: int = 8) -> InlineKeyboardMarkup:
    """
    Creates inline buttons: "Watch <KEYCODE>".
    Each press triggers callback_data: "watch:<KEYCODE>".
    """
    norm = []
    seen = set()
    for k in keycodes:
        nk = normalize_keycode(k)
        if nk and nk not in seen:
            seen.add(nk)
            norm.append(nk)

    norm = norm[:max_buttons]
    rows = [[InlineKeyboardButton(text=f"Watch {k}", callback_data=f"watch:{k}")] for k in norm]
    return InlineKeyboardMarkup(rows) if rows else InlineKeyboardMarkup([])



def build_identify_keyboard(candidates: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for k in candidates:
        rows.append([
            InlineKeyboardButton(text=f"Watch {k}", callback_data=f"watch:{k}"),
            InlineKeyboardButton(text="Watch + nickname", callback_data=f"watchnick:{k}"),
        ])
    return InlineKeyboardMarkup(rows)

async def download_best_photo_bytes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bytes:
    """
    Downloads the highest-resolution photo from update.message.photo.
    Returns bytes.
    """
    if not update.message or not update.message.photo:
        raise ValueError("No photo found on message.")

    photo = update.message.photo[-1]  # best quality
    tg_file = await context.bot.get_file(photo.file_id)
    bio = io.BytesIO()
    await tg_file.download_to_memory(out=bio)
    return bio.getvalue()

async def handle_private_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
DM flow (Step 3: user-card detection + embedding retrieval):

  - User sends a photo to the bot in a private chat.
  - Bot downloads the image bytes.
  - Bot runs the user-card YOLO cropper (OBB detection):
      * Detects one or more Pokémon cards in the image.
      * Converts oriented bounding boxes (OBB) to axis-aligned crops (AABB).
      * Keeps the highest-confidence / largest detections (configurable limit).
      * Falls back to treating the whole image as a single card
        if no detection is found but the aspect ratio matches a card.
  - For each resulting crop:
      * Bot computes a 512-D embedding using the ResNet18 encoder.
      * Bot retrieves the top-3 most similar cards from the database
        using cosine similarity (dot product on L2-normalized vectors).
      * Bot sends the corresponding card images with similarity scores.
  - Bot sends a summary message and inline “Watch <keycode>” buttons
    (deduplicated across all crops).

Scope:
  - This pipeline runs ONLY in private chats.
  - Group chats are explicitly ignored by this handler.

Requires (loaded once at startup in post_init):

  context.application.bot_data["db"]
      -> aiosqlite.Connection
         SQLite database with cards table and stored embeddings.

  context.application.bot_data["encoder"]
      -> torch.nn.Module
         ResNet18-based encoder (final fc replaced by Identity),
         loaded from encoder/last.pt.

  context.application.bot_data["device"]
      -> torch.device
         Device used for embedding inference (cpu or cuda).

  context.application.bot_data["emb_matrix"]
      -> np.ndarray, shape (N, 512), dtype float32
         L2-normalized embeddings for all cards in the database.

  context.application.bot_data["emb_keycodes"]
      -> list[str], length N
         Keycodes aligned with rows of emb_matrix.

  context.application.bot_data["user_cropper"]
      -> ultralytics.YOLO
         YOLO model trained for Pokémon card detection in user images
         (oriented bounding boxes).

  context.application.bot_data["user_cropper_conf"]
      -> float
         Confidence threshold for user-card detection.

  context.application.bot_data["user_cropper_max_crops"]
      -> int
         Maximum number of card crops processed per image.

Notes:
  - All embeddings (database and runtime) MUST be produced with the same
    encoder checkpoint and the same preprocessing (224×224 resize +
    ImageNet normalization).
  - Cropping is a prerequisite for reliable similarity; raw full-image
    embeddings are no longer used in this flow.
"""

    if not update.message or not update.effective_chat:
        return

    # Safety: only run this pipeline in private chats
    if update.effective_chat.type != "private":
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]

    # 1) Download user image
    try:
        image_bytes = await download_best_photo_bytes(update, context)
    except Exception as e:
        logger.exception("Failed to download DM photo: %s", e)
        await update.message.reply_text("I couldn't download that image. Please try again.")
        return

    # 2) Crop using YOLO OBB (Step 3)
    user_cropper = context.application.bot_data.get("user_cropper")
    if user_cropper is None:
        await update.message.reply_text("Cropper model not loaded on the bot. Check post_init().")
        return

    conf = float(context.application.bot_data.get("user_cropper_conf", 0.60))
    max_crops = int(context.application.bot_data.get("user_cropper_max_crops", 3))

    try:
        # Run blocking cropper off the event loop
        crop_bytes_list, total_found = await asyncio.to_thread(
            _crop_user_cards_obb_to_aabb,
            user_cropper=user_cropper,
            image_bytes=image_bytes,
            conf=conf,
            max_crops=max_crops,
        )
    except Exception as e:
        logger.exception("User cropper failed: %s", e)
        await update.message.reply_text("I couldn't detect/crop the card in that photo. Please try another one.")
        return

    if not crop_bytes_list:
        await update.message.reply_text(
            "No card detected. Try a clearer photo (less glare, more straight-on, card fills more of the frame)."
        )
        return

    if total_found > len(crop_bytes_list):
        await update.message.reply_text(
            f"I detected {total_found} card(s), but I will process the best {len(crop_bytes_list)} to avoid spamming."
        )

    def looks_like_telegram_file_id(s: str | None) -> bool:
        if not s:
            return False
        s = s.strip()
        return (len(s) > 20) and ("://" not in s) and ("/" not in s)

    # 3) For each crop: embedding-based retrieval (Step 2 classifier, but on the crop)
    # Collect all candidates across crops for a final keyboard.
    all_keycodes_for_buttons: List[str] = []
    summary_lines: List[str] = []

    for crop_idx, crop_bytes in enumerate(crop_bytes_list, start=1):
        try:
            ranked = await classify_image(db, context, crop_bytes, top_k=3)
        except Exception as e:
            logger.exception("Embedding classifier failed on crop %d: %s", crop_idx, e)
            summary_lines.append(f"Crop {crop_idx}: classification failed.")
            continue

        # Normalize + unique
        candidates: list[tuple[str, float]] = []
        seen: set[str] = set()
        for keycode, sim in ranked:
            k = normalize_keycode(keycode)
            if k and k not in seen:
                seen.add(k)
                candidates.append((k, float(sim)))
            if len(candidates) == 3:
                break

        if not candidates:
            summary_lines.append(f"Crop {crop_idx}: no matches.")
            continue

        # Send candidate images for this crop (same mechanism you already use)
        keycodes = [k for k, _ in candidates]
        all_keycodes_for_buttons.extend(keycodes)

        cards = await db_get_cards_by_keycodes(db, keycodes)

        upload_tasks: list[tuple[str, str, object, bool]] = []
        open_files: list[object] = []

        for rank_i, (k, sim) in enumerate(candidates, start=1):
            row = cards.get(k)
            if not row:
                continue

            _, name, image_path, image_url, telegram_file_id, *_ = row

            caption = f"Crop {crop_idx} — {rank_i}. {k} ({sim*100:.1f}%)"
            if name:
                caption += f"\n{name}"

            if looks_like_telegram_file_id(telegram_file_id):
                upload_tasks.append((k, caption, telegram_file_id, False))
                continue

            if image_path and str(image_path).strip():
                p = Path(str(image_path).strip()).expanduser()
                if p.is_file():
                    f = p.open("rb")
                    open_files.append(f)
                    upload_tasks.append((k, caption, InputFile(f, filename=p.name), True))
                    continue

            if image_url and str(image_url).strip():
                upload_tasks.append((k, caption, image_url, True))
                continue

        try:
            for k, caption, photo_arg, should_cache in upload_tasks:
                sent = await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=photo_arg,
                    caption=caption,
                )
                if should_cache and sent.photo:
                    fid = sent.photo[-1].file_id
                    await db_set_card_telegram_file_id(db, k, fid)
        finally:
            for f in open_files:
                try:
                    f.close()
                except Exception:
                    pass

        # Summary for this crop
        summary_lines.append(
            "Crop {} top matches:\n{}".format(
                crop_idx,
                "\n".join([f"- `{k}` ({sim*100:.1f}%)" for (k, sim) in candidates]),
            )
        )

    # 4) Final message + watch buttons (dedup)
    # (keep it compact; Telegram inline keyboards can get unwieldy)
    dedup_buttons: List[str] = []
    seen_btn: set[str] = set()
    for k in all_keycodes_for_buttons:
        kk = normalize_keycode(k)
        if kk and kk not in seen_btn:
            seen_btn.add(kk)
            dedup_buttons.append(kk)

    if not summary_lines:
        await update.message.reply_text("No matches found (classifier failed on all crops).")
        return

    keyboard_rows = [[InlineKeyboardButton(f"Watch {k}", callback_data=f"watch:{k}")] for k in dedup_buttons[:8]]

    await update.message.reply_text(
        "\n\n".join(summary_lines),
        reply_markup=InlineKeyboardMarkup(keyboard_rows),
        parse_mode="Markdown",
    )

# Group photo handler

async def handle_group_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat:
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]

    # Download image
    try:
        image_bytes = await download_best_photo_bytes(update, context)
    except Exception as e:
        logger.exception("Failed to download group photo: %s", e)
        return

    # Classify
    try:
        ranked = await classify_image(db, context, image_bytes, top_k=3)
    except Exception as e:
        logger.exception("Classifier failed: %s", e)
        return

    keycodes: list[str] = []
    seen: set[str] = set()
    for k, _ in ranked:
        kk = normalize_keycode(k)
        if kk and kk not in seen:
            seen.add(kk)
            keycodes.append(kk)

    if not keycodes:
        return

    watchers_by_code = await watchers_for_keycodes(db, keycodes)
    if not watchers_by_code:
        return

    # Notify watchers
    chat_name = update.effective_chat.title or "this group"

    for keycode, user_ids in watchers_by_code.items():
        for user_id in user_ids:
            try:
                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"Watched card detected in {chat_name}: {keycode}",
                )
            except TelegramError:
                pass

# Callback handler for inline buttons
async def handle_watch_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data or not query.data.startswith("watch:"):
        return

    await query.answer()  # acknowledges the button press (stops the loading spinner)

    user = query.from_user
    if not user:
        return

    keycode = normalize_keycode(query.data.split("watch:", 1)[1])
    if not keycode:
        if query.message:
            await query.message.reply_text("Invalid keycode.")
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]

    try:
        await db_add_watch(db, user.id, keycode)  # or add_watch(...) depending on your import name
    except sqlite3.Error:
        if query.message:
            await query.message.reply_text("Database error while adding to watchlist.")
        return

    text = f"Added `{keycode}` to your watchlist."

    # If the button was clicked in a group, DM the user to avoid noise/leaks.
    chat = query.message.chat if query.message else None
    if chat and chat.type != "private":
        try:
            await context.bot.send_message(chat_id=user.id, text=text, parse_mode="Markdown")
            if query.message:
                await query.message.reply_text("Done — I sent you a DM confirmation.")
        except (Forbidden, BadRequest):
            if query.message:
                await query.message.reply_text(
                    "Added, but I can't DM you yet. Please open a private chat with me and press Start."
                )
        except TelegramError:
            if query.message:
                await query.message.reply_text("Added, but failed to send DM confirmation.")
        return

    # Private chat: just reply in the same chat
    if query.message:
        await query.message.reply_text(text, parse_mode="Markdown")


async def handle_watchnick_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data or not query.data.startswith("watchnick:"):
        return

    await query.answer()

    user = update.effective_user
    if not user:
        return

    keycode = normalize_keycode(query.data.split("watchnick:", 1)[1])
    if not keycode:
        await query.message.reply_text("Invalid keycode.")
        return

    # store pending state
    context.user_data["pending_nickname_keycode"] = keycode

    # Provide skip/cancel buttons (optional but recommended)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Skip nickname", callback_data=f"watchnick_skip:{keycode}")],
        [InlineKeyboardButton("Cancel", callback_data="watchnick_cancel")],
    ])

    await query.message.reply_text(
        f"Send the nickname for `{keycode}` (max 32 chars), or tap Skip/Cancel.",
        parse_mode="Markdown",
        reply_markup=kb,
    )

async def handle_watchnick_aux_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    await query.answer()

    user = update.effective_user
    if not user:
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]

    if query.data == "watchnick_cancel":
        context.user_data.pop("pending_nickname_keycode", None)
        await query.message.reply_text("Cancelled.")
        return

    if query.data.startswith("watchnick_skip:"):
        keycode = normalize_keycode(query.data.split("watchnick_skip:", 1)[1])
        context.user_data.pop("pending_nickname_keycode", None)
        await add_watch(db, user.id, keycode, nickname=None)
        await query.message.reply_text(f"Added `{keycode}` to your watchlist.", parse_mode="Markdown")

async def handle_nickname_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return

    keycode = context.user_data.get("pending_nickname_keycode")
    if not keycode:
        return  # not in nickname flow; let other handlers process

    nickname = (update.message.text or "").strip()
    if not nickname:
        await update.message.reply_text("Nickname cannot be empty. Send a nickname, or tap Cancel.")
        return

    if nickname.lower() in ("skip", "/skip"):
        db: aiosqlite.Connection = context.application.bot_data["db"]
        await add_watch(db, update.effective_user.id, keycode, nickname=None)
        context.user_data.pop("pending_nickname_keycode", None)
        await update.message.reply_text(f"Added `{keycode}` to your watchlist.", parse_mode="Markdown")
        return

    if len(nickname) > 32:
        await update.message.reply_text("Nickname too long (max 32). Try again, or tap Cancel.")
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]
    try:
        await add_watch(db, update.effective_user.id, keycode, nickname=nickname)
    except sqlite3.IntegrityError:
        await update.message.reply_text(
            "That nickname is already used in your watchlist. Send a different nickname, or tap Cancel."
        )
        return

    context.user_data.pop("pending_nickname_keycode", None)
    await update.message.reply_text(
        f"Added `{keycode}` as *{nickname}*.",
        parse_mode="Markdown",
    )


# Commands

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! I am your Pokemon Card Tracker Bot. Use /help to see what I can do!')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Available commands:\n'
    '/start - Start the bot\n'
    '/help - Show this help message\n'
    '/watch <keycode> [nickname] - Add a card to your watchlist\n'
    '/unwatch <keycode|nickname> - Remove a card from your watchlist\n'
    '/identify - Send a photo of a card to identify its keycode\n'
    '/watchlist - Show your current watchlist\n'
    '/clearwatchlist - Remove all cards from your watchlist (DM only)\n'
    )

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('This is a custom command response!')

async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_user:
        return

    if not context.args:
        await update.message.reply_text("Usage: /watch <keycode> [nickname]")
        return

    keycode = normalize_keycode(context.args[0])
    nickname = " ".join(context.args[1:]).strip() if len(context.args) > 1 else None

    if nickname and len(nickname) > 32:
        await update.message.reply_text("Nickname too long (max 32 characters).")
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]

    try:
        await db_add_watch(db, update.effective_user.id, keycode, nickname=nickname)
    except sqlite3.IntegrityError:
        await update.message.reply_text("Nickname already used in your watchlist. Pick another one.")
        return

    if nickname:
        await update.message.reply_text(f"Watching {keycode} as '{nickname}'.")
    else:
        await update.message.reply_text(f"Watching {keycode}.")


async def unwatch_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_user:
        return

    if not context.args:
        await update.message.reply_text("Usage: /unwatch <keycode|nickname>")
        return

    token = " ".join(context.args).strip()
    db: aiosqlite.Connection = context.application.bot_data["db"]
    user_id = update.effective_user.id

    # Prefer treating it as keycode first; if nothing removed, try nickname.
    removed = await db_remove_watch(db, user_id, token)
    if removed == 0:
        removed = await db_remove_watch_by_nickname(db, user_id, token)

    if removed == 0:
        await update.message.reply_text("Nothing removed. Check the keycode/nickname.")
    else:
        await update.message.reply_text("Removed from your watchlist.")


async def identify_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    chat = update.effective_chat
    if chat and chat.type != "private":
        await update.message.reply_text("Please DM me /identify, then send the card photo.")
        return

    # Your existing DM photo handler does the actual classification. :contentReference[oaicite:5]{index=5}
    await update.message.reply_text(
        "Send me a clear photo of the card here, and I'll reply with the detected keycode(s)."
    )

async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not update.message:
        return

    user_id = update.effective_user.id
    chat = update.effective_chat

    db: aiosqlite.Connection = context.application.bot_data["db"]
    cur = await db.execute(
        "SELECT keycode, nickname FROM watchlist WHERE user_id = ? ORDER BY keycode",
        (user_id,),
    )
    rows = await cur.fetchall()

    if not rows:
        text = (
            "Your watchlist is empty. DM me a card photo to add one "
            "(send a photo in a private chat and tap “Watch ...”)."
        )
    else:
        max_show = 50
        shown = rows[:max_show]

        lines = ["Your watchlist:"]
        for keycode, nickname in shown:
            if nickname:
                lines.append(f"- `{keycode}` — *{nickname}*")
            else:
                lines.append(f"- `{keycode}`")

        if len(rows) > max_show:
            lines.append(f"(+{len(rows) - max_show} more not shown)")

        text = "\n".join(lines)

    # If invoked in a group, prefer DM to avoid spamming/leaking lists.
    if chat and chat.type != "private":
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=text,
                parse_mode="Markdown",
            )
            await update.message.reply_text("I sent you your watchlist in DM.")
        except (Forbidden, BadRequest):
            await update.message.reply_text(
                "I can't DM you yet. Please open a private chat with me, press Start, then try /watchlist again."
            )
        except TelegramError:
            await update.message.reply_text("Failed to retrieve your watchlist due to a Telegram error.")
        return

    await update.message.reply_text(text, parse_mode="Markdown")

async def clearwatchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not update.message:
        return

    chat = update.effective_chat
    user_id = update.effective_user.id

    # Avoid running destructive ops from group chats
    if chat and chat.type != "private":
        await update.message.reply_text("Please DM me /clearwatchlist to clear your personal watchlist.")
        return

    # Safety confirmation
    token = (context.args[0].lower() if context.args else "")
    if token not in ("confirm", "yes"):
        await update.message.reply_text(
            "This will remove ALL cards from your watchlist.\n"
            "To confirm, run:\n"
            "/clearwatchlist confirm"
        )
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]
    removed = await db_clear_watchlist(db, user_id)
    await update.message.reply_text(f"Cleared your watchlist. Removed {removed} entr(y/ies).")


# Responses

def handle_response(txt: str) -> str:

    processed_txt = txt.lower()
    if 'hello' in processed_txt:
        return 'Hello there!'
    return "I didn't understand that. Type /help for assistance."

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User: {update.message.chat.id} in {message_type} : "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        response: str = handle_response(text)

    print(f'Bot: "{response}"')
    await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

async def post_init(app: Application):
    db = await init_db(DB_PATH)
    app.bot_data["db"] = db

    # ---- Load encoder ----
    device = torch.device("cpu")
    encoder_path = ENCODER_PATH
    encoder = build_resnet18_encoder_from_lastpt(encoder_path, device)

    app.bot_data["encoder"] = encoder
    app.bot_data["device"] = device

    # ---- Load card embeddings into memory ----
    rows = await db.execute_fetchall(
        "SELECT keycode, embedding FROM cards WHERE embedding IS NOT NULL"
    )

    keycodes = []
    vectors = []

    for keycode, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.shape[0] != 512:
            continue
        keycodes.append(str(keycode))
        vectors.append(vec)

    if not vectors:
        raise RuntimeError("No embeddings found in DB. Did you run backfill?")

    mat = np.vstack(vectors)  # shape (N, 512)

    app.bot_data["emb_keycodes"] = keycodes
    app.bot_data["emb_matrix"] = mat

    print(f"Loaded {len(keycodes)} card embeddings into memory")
    # Step 3: YOLO user cropper (load once)
    app.bot_data["user_cropper"] = YOLO(USER_CROPPER_MODEL_PATH)
    app.bot_data["user_cropper_conf"] = USER_CROPPER_CONF
    app.bot_data["user_cropper_max_crops"] = USER_CROPPER_MAX_CROPS
    await app.bot.set_my_commands([
        BotCommand("start", "Start the bot"),
        BotCommand("help", "Show help"),
        BotCommand("identify", "Identify a card from a photo (DM only)"),
        BotCommand("watch", "Add a card to your watchlist"),
        BotCommand("unwatch", "Remove a card from your watchlist"),
        BotCommand("watchlist", "Show your watchlist"),
        BotCommand("clearwatchlist", "Remove all cards from your watchlist (DM)")

    ])






if __name__ == '__main__':
    

    # init sqlite and store handle
    app = Application.builder().token(TOKEN).local_mode(False).post_init(post_init).build()   
    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))
    app.add_handler(CommandHandler('watch', watch_command))
    app.add_handler(CommandHandler('unwatch', unwatch_command))
    app.add_handler(CommandHandler('identify', identify_command))
    app.add_handler(CommandHandler('watchlist', watchlist_command))
    app.add_handler(CommandHandler("clearwatchlist", clearwatchlist_command))

    # Photo Handlers
    app.add_handler(MessageHandler(filters.PHOTO & filters.ChatType.PRIVATE, handle_private_photo))
    app.add_handler(MessageHandler(filters.PHOTO & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP), handle_group_photo))
 
    app.add_handler(CallbackQueryHandler(handle_watch_callback, pattern=r"^watch:"))
    app.add_handler(CallbackQueryHandler(handle_watchnick_callback, pattern=r"^watchnick:"))
    app.add_handler(CallbackQueryHandler(handle_watchnick_aux_callback, pattern=r"^watchnick_skip:|^watchnick_cancel$"))

    # Group 0: stateful nickname replies (non-blocking)
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
            handle_nickname_reply,
            block=False,
        ),
        group=0,
    )

    # Group 1: normal text handling ("hello", fallback, etc.)
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        ),
        group=1,
    )

    # Errors
    app.add_error_handler(error)


    # Run the bot
    print('Bot is running...')
    app.run_polling(poll_interval=3)