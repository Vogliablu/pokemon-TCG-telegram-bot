import os
import io
import asyncio
import tempfile
import logging
import time
from dotenv import load_dotenv
from typing import Final,Iterable, List, Set, Dict, Tuple, Optional
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand,
    InputFile, InputMediaPhoto, User
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
    ensure_user,
    create_pending_prototype,
    get_latest_pending_prototype,
    delete_pending_prototype,
    create_user_prototype_from_pending,
    delete_user_prototype_by_nickname,
    set_user_prototype_threshold,
    upsert_single_pending_prototype,
    get_user_prototype_by_nickname
)
import uuid
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
USER_UPLOADS_DIR = Path("data/user_uploads")  # keep out of git
USER_CROPPER_MODEL_PATH = VISION_ROOT / "user_cropper" / "weights" / "best.pt"
USER_CROPPER_CONF = float(os.getenv("USER_CROPPER_CONF", "0.60"))
USER_CROPPER_MAX_CROPS = int(os.getenv("USER_CROPPER_MAX_CROPS", "32"))
ENCODER_PATH = VISION_ROOT / "encoder" / "last.pt"
USER_CROPPER_PATH = VISION_ROOT / "user_cropper" / "weights" / "best.pt"
TEL_CROPPER_PATH  = VISION_ROOT / "tel_cropper" / "weights" / "best.pt"

TEL_CROPPER_MODEL_PATH = str(Path("vision") / "tel_cropper" / "weights" / "best.pt")

WATCH_SIM_THRESHOLD = float(os.getenv("WATCH_SIM_THRESHOLD", "0.73"))  # global default threshold for watchers
TEL_CROPPER_CONF = float(os.getenv("TEL_CROPPER_CONF", "0.25"))
TEL_CROPPER_IMGSZ = int(os.getenv("TEL_CROPPER_IMGSZ", "960"))
TEL_CROPPER_PAD = float(os.getenv("TEL_CROPPER_PAD", "0.01"))
TEL_CROPPER_MAX_CROPS = int(os.getenv("TEL_CROPPER_MAX_CROPS", "32"))
WATCH_MAX_MATCHES_PER_MESSAGE = int(os.getenv("WATCH_MAX_MATCHES_PER_MESSAGE", "3")) # how many per message to notify about

EMBED_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])




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
def command_is_addressed_to_me(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    msg = update.effective_message
    if not msg or not msg.text:
        return False

    # Command entity at the start of the message
    ent = None
    for e in (msg.entities or []):
        if e.type == "bot_command" and e.offset == 0:
            ent = e
            break
    if not ent:
        return False

    cmd_text = msg.text[ent.offset : ent.offset + ent.length]  # e.g. "/help@MyBot" or "/help"
    if "@" not in cmd_text:
        # In groups, require explicit @mention
        return msg.chat.type == "private"

    _, at = cmd_text.split("@", 1)
    return at.lower() == (context.bot.username or "").lower()

def message_is_addressed_to_me(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    True if:
      - private chat: always True
      - group/supergroup: only True when command is like /cmd@MyBot
    """
    chat = update.effective_chat
    msg = update.effective_message
    if not chat or not msg or not msg.text:
        return False

    if chat.type == "private":
        return True

    if chat.type not in ("group", "supergroup"):
        return False

    # Find a bot_command entity at the beginning of the message
    ent = None
    for e in (msg.entities or []):
        if e.type == "bot_command" and e.offset == 0:
            ent = e
            break
    if ent is None:
        return False

    cmd_text = msg.text[ent.offset : ent.offset + ent.length]  # e.g. "/help@MyBot" or "/help"
    if "@" not in cmd_text:
        # In groups, require explicit @mention
        return False

    _, at = cmd_text.split("@", 1)
    at = at.strip().lower()

    my_username = context.application.bot_data.get("bot_username", "")
    return bool(my_username) and at == my_username

async def reply_privately(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    *,
    parse_mode: str | None = None,
) -> None:
    chat = update.effective_chat
    if chat and chat.type == "private" and update.message:
        await update.message.reply_text(text, parse_mode=parse_mode)
    else:
        # group/supergroup (or unknown): DM only, silent on failure
        ok = await send_dm_only(update, context, text, parse_mode=parse_mode)
        if not ok and update.effective_user:
            logger.info("Cannot DM user %s (hasn't started bot?)", update.effective_user.id)



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


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _image_looks_like_single_card(img: Image.Image, tol: float = 0.18) -> bool:
    w, h = img.size
    if w < 20 or h < 20:
        return False
    r = w / h
    rp = CARD_RATIO_PORTRAIT
    rl = 1.0 / rp
    return (abs(r - rp) <= tol) or (abs(r - rl) <= tol)


def _expand_box_to_aspect(xyxy, img_w: int, img_h: int, ratio_portrait: float = CARD_RATIO_PORTRAIT):
    x1, y1, x2, y2 = map(float, xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    cur = bw / bh
    rp = ratio_portrait
    rl = 1.0 / rp
    target = rp if abs(cur - rp) <= abs(cur - rl) else rl

    # Expand (never shrink) to reach target ratio
    desired_w, desired_h = bw, bh
    if cur < target:
        desired_w = max(bw, bh * target)
    else:
        desired_h = max(bh, bw / target)

    nx1 = cx - desired_w / 2.0
    nx2 = cx + desired_w / 2.0
    ny1 = cy - desired_h / 2.0
    ny2 = cy + desired_h / 2.0

    nx1 = _clamp(nx1, 0.0, img_w - 2.0)
    ny1 = _clamp(ny1, 0.0, img_h - 2.0)
    nx2 = _clamp(nx2, nx1 + 1.0, float(img_w))
    ny2 = _clamp(ny2, ny1 + 1.0, float(img_h))
    return [nx1, ny1, nx2, ny2]

def _crop_with_pad(img: Image.Image, xyxy, pad_frac: float) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = map(float, xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = pad_frac * bw
    py = pad_frac * bh
    x1 = _clamp(int(round(x1 - px)), 0, w - 1)
    y1 = _clamp(int(round(y1 - py)), 0, h - 1)
    x2 = _clamp(int(round(x2 + px)), x1 + 1, w)
    y2 = _clamp(int(round(y2 + py)), y1 + 1, h)
    return img.crop((x1, y1, x2, y2))

def _rotate_cw_if_landscape(crop: Image.Image) -> Image.Image:
    return crop.rotate(-90, expand=True) if crop.width > crop.height else crop

def _rotate_ccw_if_landscape(crop: Image.Image) -> Image.Image:
    return crop.rotate(90, expand=True) if crop.width > crop.height else crop

def _pil_open_rgb_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGBA").convert("RGB")

def crop_telegram_cards_from_bytes(
    *,
    tel_cropper,
    image_bytes: bytes,
    conf: float,
    imgsz: int,
    pad: float,
    max_crops_per_image: int,
    aspect_expand: bool = True,
    scan_fallback: bool = True,
    rotate_portrait: bool = True,
) -> list[bytes]:
    """
    Returns list of crop PNG bytes.
    Mirrors crop_telegram_pictures.py behavior (boxes.xyxy + aspect expand + scan fallback + pad + optional dual rotations).
    """
    img = _pil_open_rgb_from_bytes(image_bytes)
    W, H = img.size

    # Ultralytics can accept PIL directly; use that to avoid temp files.
    results = tel_cropper.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)
    boxes = results[0].boxes if results else None

    crops: list[bytes] = []

    if boxes is None or len(boxes) == 0:
        if scan_fallback and _image_looks_like_single_card(img):
            # Full-scan fallback. If landscape and rotate_portrait, emit both rotations.
            if rotate_portrait and img.width > img.height:
                for im2 in (_rotate_cw_if_landscape(img), _rotate_ccw_if_landscape(img)):
                    bio = io.BytesIO()
                    im2.save(bio, format="PNG")
                    crops.append(bio.getvalue())
            else:
                bio = io.BytesIO()
                img.save(bio, format="PNG")
                crops.append(bio.getvalue())
        return crops

    xyxy_all = boxes.xyxy.cpu().numpy()
    conf_all = boxes.conf.cpu().numpy()

    # Sort by confidence desc
    order = np.argsort(conf_all)[::-1]
    if max_crops_per_image and max_crops_per_image > 0:
        order = order[:max_crops_per_image]

    for idx in order.tolist():
        xyxy = xyxy_all[idx].tolist()
        if aspect_expand:
            xyxy = _expand_box_to_aspect(xyxy, W, H, ratio_portrait=CARD_RATIO_PORTRAIT)
        crop = _crop_with_pad(img, xyxy, pad_frac=pad)

        # If landscape and rotate_portrait enabled -> emit both rotations
        if rotate_portrait and crop.width > crop.height:
            for c2 in (_rotate_cw_if_landscape(crop), _rotate_ccw_if_landscape(crop)):
                bio = io.BytesIO()
                c2.save(bio, format="PNG")
                crops.append(bio.getvalue())
        else:
            bio = io.BytesIO()
            crop.save(bio, format="PNG")
            crops.append(bio.getvalue())

    return crops


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

async def rebuild_user_prototypes_cache(db: aiosqlite.Connection) -> dict[int, dict[str, object]]:
    """
    Returns:
      cache[user_id] = {
        "ids": [int,...],
        "nicks": [str,...],
        "mat": np.ndarray shape (M, 512) float32 L2-normalized,
        "ths": [float,...]
      }
    """
    cur = await db.execute(
        """
        SELECT id, owner_user_id, nickname, threshold, embedding
        FROM user_prototypes
        WHERE embedding IS NOT NULL
        """
    )
    rows = await cur.fetchall()

    tmp: dict[int, dict[str, list]] = {}
    for proto_id, owner_user_id, nickname, threshold, emb_blob in rows:
        uid = int(owner_user_id)
        v = np.frombuffer(emb_blob, dtype=np.float32)
        if v.shape[0] != 512:
            continue
        d = tmp.setdefault(uid, {"ids": [], "nicks": [], "ths": [], "vecs": []})
        d["ids"].append(int(proto_id))
        d["nicks"].append(str(nickname))
        d["ths"].append(float(threshold) if threshold is not None else 0.73)
        d["vecs"].append(v)

    cache: dict[int, dict[str, object]] = {}
    for uid, d in tmp.items():
        mat = np.vstack(d["vecs"]).astype(np.float32, copy=False)
        ths = np.array(d["ths"], dtype=np.float32)
        cache[uid] = {"ids": d["ids"], "nicks": d["nicks"], "ths": ths, "mat": mat}
    return cache



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


@torch.no_grad()
def compute_embedding_from_png_or_jpg_bytes(
    image_bytes: bytes,
    *,
    encoder: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        x = EMBED_TF(im).unsqueeze(0).to(device)
    v = encoder(x).detach().cpu().numpy().astype(np.float32)[0]  # (512,)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


async def send_dm_only(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    *,
    parse_mode: str | None = None,
) -> bool:
    """
    Sends a DM to the user. If invoked from a group, sends ONLY the DM and stays silent in the group.
    Returns True if the DM was sent, False otherwise.
    """
    if not update.effective_user:
        return False

    user_id = update.effective_user.id
    try:
        await context.bot.send_message(chat_id=user_id, text=text, parse_mode=parse_mode)
        return True
    except (Forbidden, BadRequest):
        # User hasn't started the bot or bot cannot DM them.
        return False
    except TelegramError:
        return False

def html_mention(user: User | None) -> str:
    if not user:
        return "Someone"
    name = user.full_name or "Someone"
    # minimal HTML escaping for safety
    name = (
        name.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )
    return f'<a href="tg://user?id={user.id}">{name}</a>'


async def handle_private_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat:
        return
    if update.effective_chat.type != "private":
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]
    user_id = update.effective_user.id

    await ensure_user(db, user_id)

    # 1) Download user image
    try:
        image_bytes = await download_best_photo_bytes(update, context)
    except Exception as e:
        logger.exception("Failed to download DM photo: %s", e)
        await update.message.reply_text("I couldn't download that image. Please try again.")
        return

    # 2) Crop using YOLO OBB
    user_cropper = context.application.bot_data.get("user_cropper")
    if user_cropper is None:
        await update.message.reply_text("Cropper model not loaded. Check post_init().")
        return

    conf = float(context.application.bot_data.get("user_cropper_conf", 0.60))

    try:
        # IMPORTANT: allow multiple detections so we can enforce "exactly one"
        crop_bytes_list, total_found = await asyncio.to_thread(
            _crop_user_cards_obb_to_aabb,
            user_cropper=user_cropper,
            image_bytes=image_bytes,
            conf=conf,
            max_crops=10,   # do not cap at 1 here; we must detect multiplicity
        )
    except Exception as e:
        logger.exception("User cropper failed: %s", e)
        await update.message.reply_text("I couldn't detect/crop the card in that photo. Please try another one.")
        return

    # 3) Enforce exactly one card
    if total_found > 1:
        await update.message.reply_text(
            f"I detected {total_found} cards in this picture. Please send a photo with one card only."
        )
        return

    if not crop_bytes_list:
        await update.message.reply_text(
            "No card detected. Try a clearer photo (less glare, more straight-on, card fills more of the frame)."
        )
        return

    # At this point we accept exactly one crop
    crop_bytes = crop_bytes_list[0]

    # 4) Compute embedding
    try:
        encoder = context.application.bot_data["encoder"]
        device = context.application.bot_data["device"]
        v = await asyncio.to_thread(
            compute_embedding_from_png_or_jpg_bytes,
            crop_bytes,
            encoder=encoder,
            device=device,
        )
    except Exception as e:
        logger.exception("Embedding computation failed: %s", e)
        await update.message.reply_text("I couldn't process that card image. Please try again.")
        return

    emb_blob = v.astype(np.float32).tobytes(order="C")

    # 5) Persist as pending + save crop to disk (Option B: keep only one pending per user)
    token = str(uuid.uuid4())

    out_dir = USER_UPLOADS_DIR / str(user_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = out_dir / f"{token}.png"
    try:
        crop_path.write_bytes(crop_bytes)
    except Exception as e:
        logger.exception("Failed to write crop to disk: %s", e)
        await update.message.reply_text("Internal error saving the crop. Please try again.")
        return

    try:
        # This replaces any previous pending for this user and returns the old image_path (if any)
        old_path = await upsert_single_pending_prototype(
            db,
            token=token,
            owner_user_id=user_id,
            image_path=str(crop_path),
            embedding_blob=emb_blob,
            embedding_dim=512,
            embedding_norm=1,
            embedding_model=context.application.bot_data.get("embedding_model_tag"),
        )
    except Exception as e:
        logger.exception("Failed to store pending prototype: %s", e)
        await update.message.reply_text("Internal error storing pending card. Please try again.")
        return

    # Best-effort: delete the previous pending crop file from disk (avoid clutter)
    try:
        if old_path and old_path != str(crop_path):
            Path(old_path).unlink(missing_ok=True)
    except Exception:
        pass

    # Convenience: store token in user_data too (DB remains source of truth)
    context.user_data["pending_token"] = token

    # 6) Send the crop back + instructions
    try:
        bio = io.BytesIO(crop_bytes)
        bio.name = "card.png"
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=InputFile(bio),
            caption=(
                "I detected one card.\n\n"
                "If you want to watch it, reply with:\n"
                "`/watch <nickname>`\n\n"
                "Example:\n"
                "`/watch charizard`\n\n"
                "Or cancel with `/cancel`.\n"
                "Only the latest pending card can be watched."
            ),
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.exception("Failed to send cropped preview: %s", e)
        await update.message.reply_text(
            "I detected one card. Reply with `/watch <nickname>` to watch it, or `/cancel`.",
            parse_mode="Markdown",
        )
# Group photo handler

async def handle_group_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat:
        return

    chat = update.effective_chat
    if chat.type not in ("group", "supergroup"):
        return

    # Ignore photos sent by bots (optional but sensible)
    if update.effective_user and update.effective_user.is_bot:
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]

    try:
        image_bytes = await download_best_photo_bytes(update, context)
    except Exception as e:
        logger.exception("Failed to download group photo: %s", e)
        return

    tel_cropper = context.application.bot_data.get("tel_cropper")
    if tel_cropper is None:
        logger.error("tel_cropper not loaded; check post_init")
        return

    conf = float(context.application.bot_data.get("tel_cropper_conf", 0.25))
    imgsz = int(context.application.bot_data.get("tel_cropper_imgsz", 960))
    pad = float(context.application.bot_data.get("tel_cropper_pad", 0.01))
    max_crops = int(context.application.bot_data.get("tel_cropper_max_crops", 6))

    # Policy: notify up to K matches per user per group message
    K = int(context.application.bot_data.get("watch_max_matches_per_message", 3))
    K = max(1, min(K, 8))  # safety cap

    # 1) Crop telegram image (blocking -> thread)
    try:
        crop_bytes_list = await asyncio.to_thread(
            crop_telegram_cards_from_bytes,
            tel_cropper=tel_cropper,
            image_bytes=image_bytes,
            conf=conf,
            imgsz=imgsz,
            pad=pad,
            max_crops_per_image=max_crops,
            aspect_expand=True,
            scan_fallback=True,
            rotate_portrait=True,
        )
    except Exception as e:
        logger.exception("Telegram cropper failed: %s", e)
        return

    if not crop_bytes_list:
        return  # nothing detected

    # 2) Compute embeddings for crops (keep alignment)
    encoder = context.application.bot_data["encoder"]
    device = context.application.bot_data["device"]

    crop_items: list[tuple[bytes, np.ndarray]] = []
    for cb in crop_bytes_list:
        try:
            v = await asyncio.to_thread(
                compute_embedding_from_png_or_jpg_bytes,
                cb,
                encoder=encoder,
                device=device,
            )
            crop_items.append((cb, v))
        except Exception:
            continue

    if not crop_items:
        return

    # 3) Ensure prototypes cache is ready
    if context.application.bot_data.get("proto_cache_dirty", True) or not context.application.bot_data.get("proto_cache"):
        try:
            context.application.bot_data["proto_cache"] = await rebuild_user_prototypes_cache(db)
            context.application.bot_data["proto_cache_dirty"] = False
            logger.info("Rebuilt user_prototypes cache")
        except Exception as e:
            logger.exception("Failed to rebuild user_prototypes cache: %s", e)
            return

    proto_cache: dict[int, dict[str, object]] = context.application.bot_data["proto_cache"]
    if not proto_cache:
        return  # no watchers

    # Useful context for DM message
    group_title = chat.title or str(chat.id)
    sender = update.effective_user if update.effective_user else "Someone"
    sender_mention = html_mention(sender)

    # Optional: show @username too (if present)
    sender_username = ""
    if sender and sender.username:
        sender_username = f" (@{sender.username})"    
    msg_id = update.message.message_id
    chat_id = chat.id

    now = time.time()
    notified_triplets: dict[tuple[int, int, int], float] = context.application.bot_data.setdefault(
        "notified_triplets", {}
    )
    ttl = int(context.application.bot_data.get("notified_ttl_seconds", 3600))
    prune_every = int(context.application.bot_data.get("notified_prune_every", 5000))

    # Prune only when structure is "large enough" (cheap amortized maintenance)
    if len(notified_triplets) > prune_every:
        cutoff = now - ttl
        for k, ts in list(notified_triplets.items()):
            if ts < cutoff:
                del notified_triplets[k]

    # 4) Compare per user and notify at most once per message (but include up to K matches)
    for user_id, d in proto_cache.items():
        uid = int(user_id)
        triplet = (chat_id, msg_id, uid)
        if triplet in notified_triplets:
            continue

        mat: np.ndarray = d["mat"]          # (M,512)
        nicks: list[str] = d["nicks"]       # length M
        ths = d.get("ths")                  # np.ndarray shape (M,) or list length M

        # Safety fallback if cache lacks thresholds
        if ths is None:
            ths = np.full((mat.shape[0],), 0.73, dtype=np.float32)
        elif not isinstance(ths, np.ndarray):
            ths = np.array(ths, dtype=np.float32)

        # Collect all passing matches; dedup by nickname keeping best similarity
        # matches[nick] = (sim, crop_bytes, threshold_used)
        matches: dict[str, tuple[float, bytes, float]] = {}

        for cb, v in crop_items:
            sims = mat @ v
            j = int(np.argmax(sims))
            s = float(sims[j])
            th_j = float(ths[j])
            nick = nicks[j]

            if s < th_j:
                continue

            prev = matches.get(nick)
            if prev is None or s > prev[0]:
                matches[nick] = (s, cb, th_j)

        if not matches:
            continue

        # Sort and keep top-K
        top = sorted(matches.items(), key=lambda kv: kv[1][0], reverse=True)[:K]

        # Mark de-dup before sending to avoid double-notifies on retry
        notified_triplets[triplet] = now

        # Send summary + up to K crop images
        summary_lines = [
        f"Match(es) found in group: {group_title}",
        f"Posted by: {sender_mention}{sender_username}",
        "",
        "Top matches:",
]
        for rank, (nick, (sim, _cb, th_used)) in enumerate(top, start=1):
            summary_lines.append(f"{rank}. {nick} — sim {sim:.3f} (th {th_used:.2f})")

        try:
            await context.bot.send_message(
                chat_id=uid,
                text="\n".join(summary_lines),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except (Forbidden, BadRequest):
            # User hasn't started bot / cannot be messaged; silently skip
            continue
        except TelegramError as e:
            logger.warning("Failed to DM summary to user %s: %s", uid, e)
            continue

        for rank, (nick, (sim, cb, th_used)) in enumerate(top, start=1):
            caption = f"{rank}/{len(top)} — {nick}\nSimilarity: {sim:.3f} (threshold {th_used:.2f})"
            try:
                bio = io.BytesIO(cb)
                bio.name = "match.png"
                await context.bot.send_photo(
                    chat_id=uid,
                    photo=InputFile(bio),
                    caption=caption,
                )
            except (Forbidden, BadRequest):
                break
            except TelegramError as e:
                logger.warning("Failed to DM user %s match photo: %s", uid, e)
                continue

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
    if not message_is_addressed_to_me(update, context):
        return
    await update.message.reply_text('Hello! I am your Pokemon Card Tracker Bot. Use /help to see what I can do!')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not message_is_addressed_to_me(update, context):
        return

    help_text = (
        "Recommended usage:\n"
        "Send a card image to me, then use /watch to save it to your watchlist.\n"
        "I will notify you when I find your card in a group.\n"
        "Text me in a private chat for commands.\n"
        "\n"
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/watch <nickname> - Add the last detected card in your watchlist\n"
        "/unwatch <nickname> - Remove a card from your watchlist\n"
        "/watchlist - Show your current watchlist\n"
        "/clearwatchlist - Remove all cards from your watchlist\n"
        "/setthreshold <nickname> <threshold> - Set similarity threshold for a watched card\n"
        "/show <nickname> - Show details for a watched card\n"
        "\n"
        "Threshold guide (similarity): \n"
        "I will notify you only when the similarity is above the threshold you set.\n"
        "Set a higher threshold to reduce false positives, or a lower threshold to catch more matches.\n"
        "(Very likely ≥ 0.77 | Quite likely 0.73–0.77 | Possible 0.68–0.73)\n"
    )

    chat = update.effective_chat
    if chat and chat.type != "private":
        # DM only; do not reply in the group
        await reply_privately(update, context, help_text)
        return

    await update.message.reply_text(help_text)


def _normalize_nickname(n: str) -> str:
    return n.strip().lower()

async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not message_is_addressed_to_me(update, context):
        return        
    
    if not update.message or not update.effective_chat:
        return
    if update.effective_chat.type != "private":
        await reply_privately(update, context, "Use /watch in a private chat with me.")
        return
    
    db: aiosqlite.Connection = context.application.bot_data["db"]
    user_id = update.effective_user.id

    if not context.args and update.effective_chat.type == "private":
        await reply_privately(update, context, "Usage: First send an image of your card, then use `/watch <nickname>`", parse_mode="Markdown")
        return

    nickname = _normalize_nickname(" ".join(context.args))
    if not nickname:
        await reply_privately(update, context, "Nickname cannot be empty.")
        return
    if len(nickname) > 40:
        await reply_privately(update, context, "Nickname is too long (max 40 chars).")
        return

    # Prefer token from user_data, but fall back to latest pending in DB
    pending = await get_latest_pending_prototype(db, owner_user_id=user_id)
    if not pending:
        await reply_privately(update, context, "No pending card found. Send me a card photo first.")
        return

    token, image_path, emb_blob, dim, norm, model = pending

    try:
        proto_id = await create_user_prototype_from_pending(
            db,
            owner_user_id=user_id,
            nickname=nickname,
            image_path=image_path,
            embedding_blob=emb_blob,
            embedding_dim=dim,
            embedding_norm=norm,
            embedding_model=model,
        )
        
    except sqlite3.IntegrityError:
        await reply_privately(update, context,
            f"You already have a watched card named `{nickname}`. Choose another nickname.",
            parse_mode="Markdown",
        )
        return
    except Exception as e:
        logger.exception("Failed to create user prototype: %s", e)
        await reply_privately(update, context, "Failed to save this watched card. Please try again.")
        return
    context.application.bot_data["proto_cache_dirty"] = True

    # Delete pending row (we keep the image_path as-is)
    try:
        await delete_pending_prototype(db, token=token, owner_user_id=user_id)
    except Exception as e:
        logger.warning("Failed to delete pending prototype %s: %s", token, e)

    context.user_data.pop("pending_token", None)

    await reply_privately(update, context,
        f"Saved. I will watch this card as `{nickname}`.",
        parse_mode="Markdown",
    )


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not message_is_addressed_to_me(update, context):
        return  
    
    if not update.message or not update.effective_chat:
        return
    if update.effective_chat.type != "private":
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]
    user_id = update.effective_user.id

    pending = await get_latest_pending_prototype(db, owner_user_id=user_id)
    if not pending:
        await reply_privately(update, context, "No pending card to cancel.")
        return

    token, image_path, *_ = pending

    try:
        await delete_pending_prototype(db, token=token, owner_user_id=user_id)
    except Exception as e:
        logger.exception("Failed to delete pending prototype: %s", e)

    # best-effort delete file
    try:
        if image_path:
            Path(image_path).unlink(missing_ok=True)
    except Exception:
        pass

    context.user_data.pop("pending_token", None)
    await reply_privately(update, context, "Cancelled. Send another card photo when ready.")

async def unwatch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not message_is_addressed_to_me(update, context):
        return    
    
    if not update.message or not update.effective_user:
        return

    user_id = update.effective_user.id
    chat = update.effective_chat
    db: aiosqlite.Connection = context.application.bot_data["db"]

    if not context.args:
        msg = "Usage: `/unwatch <nickname>`"
        if chat and chat.type != "private":
            # prefer DM
            try:
                await context.bot.send_message(chat_id=user_id, text=msg, parse_mode="Markdown")
                #await update.message.reply_text("I sent you instructions in DM.")
            except Exception:
                await reply_privately(update, context, msg, parse_mode="Markdown")
            return
        await reply_privately(update, context, msg, parse_mode="Markdown")
        return

    nickname = _normalize_nickname(" ".join(context.args))
    if not nickname:
        await reply_privately(update, context, "Nickname cannot be empty.")
        return

    try:
        image_path = await delete_user_prototype_by_nickname(
            db, owner_user_id=user_id, nickname=nickname
        )
    except Exception as e:
        logger.exception("Failed to delete user prototype: %s", e)
        await reply_privately(update, context, "Failed to remove that card. Please try again.")
        return

    if image_path is None:
        text = f"No learned card named `{nickname}` found."
    else:
        # Best-effort file cleanup
        context.application.bot_data["proto_cache_dirty"] = True

        try:
            if image_path:
                Path(image_path).unlink(missing_ok=True)
        except Exception:
            pass
        text = f"Removed learned card `{nickname}`."

    # If invoked outside private: DM result and stay silent in the group
    if chat and chat.type != "private":
        try:
            await context.bot.send_message(chat_id=user_id, text=text, parse_mode="Markdown")
        except (Forbidden, BadRequest):
            # Can't DM user (likely hasn't started the bot). Stay silent in group.
            logger.info("Cannot DM user %s (hasn't started bot?) for /unwatch result.", user_id)
        except TelegramError as e:
            logger.warning("Telegram error while DMing user %s: %s", user_id, e)
        return

    await reply_privately(update, context, text, parse_mode="Markdown")

# async def identify_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     if not update.message:
#         return

#     chat = update.effective_chat
#     if chat and chat.type != "private":
#         await update.message.reply_text("Please DM me /identify, then send the card photo.")
#         return

#     # Your existing DM photo handler does the actual classification. :contentReference[oaicite:5]{index=5}
#     await update.message.reply_text(
#         "Send me a clear photo of the card here, and I'll reply with the detected keycode(s)."
#     )

async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not message_is_addressed_to_me(update, context):
        return    
    
    if not update.effective_user or not update.message:
        return

    user_id = update.effective_user.id
    chat = update.effective_chat

    db: aiosqlite.Connection = context.application.bot_data["db"]

    # Old keycode-based watchlist entries (if you still use them)
    cur = await db.execute(
        "SELECT keycode, nickname FROM watchlist WHERE user_id = ? ORDER BY keycode",
        (user_id,),
    )
    keycode_rows = await cur.fetchall()

    # New user-learned watched cards (this is what /watch <nickname> creates)
    cur = await db.execute(
        """
        SELECT id, nickname, threshold, created_at
        FROM user_prototypes
        WHERE owner_user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    )
    proto_rows = await cur.fetchall()

    if not keycode_rows and not proto_rows:
        text = (
            "Your watchlist is empty.\n\n"
            "Send me a card photo in DM. If exactly one card is detected, I will ask you to run:\n"
            "`/watch <nickname>`"
        )
    else:
        lines = []

        if proto_rows:
            lines.append("Your learned cards:")
            max_show = 50
            shown = proto_rows[:max_show]
            for proto_id, nickname, th, created_at in shown:
            # th can be NULL if you have old rows; fall back to default
                th_val = float(th) if th is not None else 0.70
                lines.append(f"- *{nickname}* - th={th_val:.2f}")

            if len(proto_rows) > max_show:
                lines.append(f"(+{len(proto_rows) - max_show} more not shown)")
            lines.append("")  # blank line between sections

        if keycode_rows:
            lines.append("Your catalog cards:")
            max_show = 50
            shown = keycode_rows[:max_show]
            for keycode, nickname in shown:
                if nickname:
                    lines.append(f"- `{keycode}` — *{nickname}*")
                else:
                    lines.append(f"- `{keycode}`")
            if len(keycode_rows) > max_show:
                lines.append(f"(+{len(keycode_rows) - max_show} more not shown)")

        text = "\n".join(lines).rstrip()

    # If invoked in a group, prefer DM to avoid spamming/leaking lists.
    if chat and chat.type != "private":
        #try:
        await context.bot.send_message(
            chat_id=user_id,
            text=text,
            parse_mode="Markdown",
        )
        #await update.message.reply_text("I sent you your watchlist in DM.")
        # except (Forbidden, BadRequest):
        #     await update.message.reply_text(
        #         "I can't DM you yet. Please open a private chat with me, press Start, then try /watchlist again."
        #     )
        # except TelegramError:
        #     await update.message.reply_text("Failed to retrieve your watchlist due to a Telegram error.")
        return

    await reply_privately(update, context, text, parse_mode="Markdown")

async def clearwatchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not message_is_addressed_to_me(update, context):
        return   
   
    if not update.effective_user or not update.message:
        return

    chat = update.effective_chat
    user_id = update.effective_user.id

    # Avoid running destructive ops from group chats
    if chat and chat.type != "private":
        await reply_privately(update, context, "Please DM me /clearwatchlist to clear your personal watchlist.")
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
    await reply_privately(update, context, f"Cleared your watchlist. Removed {removed} entr(y/ies).")

async def setthreshold_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not message_is_addressed_to_me(update, context):
        return
    
    if not update.message or not update.effective_chat:
        return
    # if update.effective_chat.type != "private":
    #     await update.message.reply_text("Use /setthreshold in a private chat with me.")
    #     return

    if len(context.args) < 2:
        await reply_privately(update, context,
            "Usage: `/setthreshold <nickname> <value>`\nExample: `/setthreshold charizard 0.86`",
            parse_mode="Markdown",
        )
        return

    nickname = _normalize_nickname(" ".join(context.args[:-1]))
    val_str = context.args[-1]

    try:
        th = float(val_str)
    except ValueError:
        await reply_privately(update, context, "Threshold must be a number like 0.86.")
        return

    if not (0.0 < th < 1.0):
        await reply_privately(update, context, "Threshold must be between 0 and 1 (e.g., 0.86).")
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]
    user_id = update.effective_user.id

    n = await set_user_prototype_threshold(db, owner_user_id=user_id, nickname=nickname, threshold=th)
    if n == 0:
        await reply_privately(update, context, f"No learned card named `{nickname}` found.", parse_mode="Markdown")
        return

    # mark cache dirty so group matching sees new threshold
    context.application.bot_data["proto_cache_dirty"] = True

    await reply_privately(
        f"Updated threshold for `{nickname}` to {th:.2f}.",
        parse_mode="Markdown",
    )

def looks_like_telegram_file_id(s: str | None) -> bool:
    if not s:
        return False
    s = s.strip()
    return (len(s) > 20) and ("://" not in s) and ("/" not in s)

async def show_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Usage: /show <nickname>
    Sends the saved prototype image for that nickname (DM only response).
    In groups, executes only if the command is explicitly addressed to the bot,
    and still replies via DM only.
    """
    if not update.effective_user or not update.effective_chat:
        return
    if not message_is_addressed_to_me(update, context):
        return

    user_id = update.effective_user.id
    db: aiosqlite.Connection = context.application.bot_data["db"]

    if not context.args:
        await reply_privately(update, context, "Usage: `/show <nickname>`", parse_mode="Markdown")
        return

    nickname = _normalize_nickname(" ".join(context.args))
    if not nickname:
        await reply_privately(update, context, "Nickname cannot be empty.")
        return

    row = await get_user_prototype_by_nickname(db, owner_user_id=user_id, nickname=nickname)
    if not row:
        await reply_privately(update, context, f"No watched card named `{nickname}` found.", parse_mode="Markdown")
        return

    proto_id, nick, image_path, telegram_file_id, threshold = row

    caption = f"{nick} (th={float(threshold):.2f})"

    # Prefer cached telegram_file_id if available
    if looks_like_telegram_file_id(telegram_file_id):
        try:
            await context.bot.send_photo(chat_id=user_id, photo=telegram_file_id, caption=caption)
            return
        except TelegramError:
            # fall back to disk if possible
            pass

    # Fallback: load from disk path
    if image_path and str(image_path).strip():
        p = Path(str(image_path)).expanduser()
        if p.is_file():
            try:
                with p.open("rb") as f:
                    await context.bot.send_photo(
                        chat_id=user_id,
                        photo=InputFile(f, filename=p.name),
                        caption=caption,
                    )
                return
            except (Forbidden, BadRequest):
                return
            except TelegramError as e:
                logger.warning("Failed to send prototype image for user %s: %s", user_id, e)

    await reply_privately(
        update,
        context,
        "I found the watched card in the database, but its image file is missing on disk.",
    )

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
    me = await app.bot.get_me()
    app.bot_data["bot_username"] = (me.username or "").lower()
    # ---- Init DB ----
    db = await init_db(DB_PATH)
    app.bot_data["db"] = db
    # initialize notified triplets storage
    app.bot_data["notified_triplets"] = {}  # (chat_id, msg_id, user_id) -> unix_ts
    app.bot_data["notified_prune_every"] = 5000
    app.bot_data["notified_ttl_seconds"] = 3600  # 1 hour

    # ---- Load encoder ----
    device = torch.device("cpu")
    encoder_path = ENCODER_PATH
    encoder = build_resnet18_encoder_from_lastpt(encoder_path, device)

    app.bot_data["encoder"] = encoder
    app.bot_data["device"] = device
    app.bot_data["tel_cropper"] = YOLO(TEL_CROPPER_MODEL_PATH)
    app.bot_data["tel_cropper_conf"] = TEL_CROPPER_CONF
    app.bot_data["tel_cropper_imgsz"] = TEL_CROPPER_IMGSZ
    app.bot_data["tel_cropper_pad"] = TEL_CROPPER_PAD
    app.bot_data["tel_cropper_max_crops"] = TEL_CROPPER_MAX_CROPS

    app.bot_data["watch_sim_threshold"] = WATCH_SIM_THRESHOLD
    app.bot_data["watch_max_matches_per_message"] = WATCH_MAX_MATCHES_PER_MESSAGE

    # watched-prototypes cache
    app.bot_data["proto_cache_dirty"] = True
    app.bot_data["proto_cache"] = {}  # filled on first use



    # # ---- Load card embeddings into memory ----
    # rows = await db.execute_fetchall(
    #     "SELECT keycode, embedding FROM cards WHERE embedding IS NOT NULL"
    # )

    # keycodes = []
    # vectors = []

    # for keycode, blob in rows:
    #     vec = np.frombuffer(blob, dtype=np.float32)
    #     if vec.shape[0] != 512:
    #         continue
    #     keycodes.append(str(keycode))
    #     vectors.append(vec)

    # if not vectors:
    #     raise RuntimeError("No embeddings found in DB. Did you run backfill?")

    # mat = np.vstack(vectors)  # shape (N, 512)

    # app.bot_data["emb_keycodes"] = keycodes
    # app.bot_data["emb_matrix"] = mat

    # print(f"Loaded {len(keycodes)} card embeddings into memory")
    # Step 3: YOLO user cropper (load once)
    app.bot_data["user_cropper"] = YOLO(USER_CROPPER_MODEL_PATH)
    app.bot_data["user_cropper_conf"] = USER_CROPPER_CONF
    app.bot_data["user_cropper_max_crops"] = USER_CROPPER_MAX_CROPS
    await app.bot.set_my_commands([
        BotCommand("start", "Start the bot"),
        BotCommand("help", "Show help"),
        BotCommand("watch", "Add a card to your watchlist, after sending a photo"),
        BotCommand("unwatch", "Remove a card from your watchlist"),
        BotCommand("watchlist", "Show your watchlist"),
        BotCommand("clearwatchlist", "Remove all cards from your watchlist"),
        BotCommand("setthreshold", "Set per-card threshold: /setthreshold <nickname> <value>"),
        BotCommand("show", "Show a watched card image by nickname")


    ])






if __name__ == '__main__':
    

    # init sqlite and store handle
    app = Application.builder().token(TOKEN).local_mode(False).post_init(post_init).build()   
    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    # app.add_handler(CommandHandler('custom', custom_command))
    app.add_handler(CommandHandler('unwatch', unwatch_command))
    #app.add_handler(CommandHandler('identify', identify_command))
    app.add_handler(CommandHandler('watchlist', watchlist_command))
    app.add_handler(CommandHandler("clearwatchlist", clearwatchlist_command))
    app.add_handler(CommandHandler("watch", watch_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("setthreshold", setthreshold_command))
    app.add_handler(CommandHandler("show", show_command))

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
    app.run_polling(
        allowed_updates=["message", "callback_query"],
        poll_interval=3,
    )
