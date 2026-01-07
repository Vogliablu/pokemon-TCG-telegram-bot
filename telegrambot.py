import os
import io
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


# Helpers
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
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Dummy classifier: nearest average RGB among all cards.
    Returns [(keycode, similarity_0_1), ...].
    """
    q_r, q_g, q_b = avg_rgb_from_bytes(image_bytes)
    index = await ensure_card_index(context, db)
    if not index:
        return []

    max_dist = (3 * (255.0 ** 2)) ** 0.5
    scored: list[tuple[str, float]] = []
    for keycode, r, g, b in index:
        dr = r - q_r
        dg = g - q_g
        dbb = b - q_b
        dist = (dr * dr + dg * dg + dbb * dbb) ** 0.5
        sim = 1.0 - (dist / max_dist)
        if sim < 0.0:
            sim = 0.0
        scored.append((keycode, sim))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]


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
    if not update.message:
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]

    # 1) Download user image
    try:
        image_bytes = await download_best_photo_bytes(update, context)
    except Exception as e:
        logger.exception("Failed to download DM photo: %s", e)
        await update.message.reply_text("I couldn't download that image. Please try again.")
        return

    # 2) Classify
    try:
        ranked = await classify_image(db, context, image_bytes, top_k=3)
    except Exception as e:
        logger.exception("Classifier failed: %s", e)
        await update.message.reply_text("Classification failed. Please try again.")
        return

    # 3) Normalize + unique
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
        await update.message.reply_text("No matches found.")
        return

    keycodes = [k for k, _ in candidates]
    cards = await db_get_cards_by_keycodes(db, keycodes)

    def looks_like_telegram_file_id(s: str | None) -> bool:
        if not s:
            return False
        s = s.strip()
        return (len(s) > 20) and ("://" not in s) and ("/" not in s)

    # 4) Build upload tasks
    upload_tasks: list[tuple[str, str, object, bool]] = []
    open_files: list[object] = []

    for idx, (k, sim) in enumerate(candidates, start=1):
        row = cards.get(k)
        if not row:
            continue

        _, name, image_path, image_url, telegram_file_id, *_ = row

        caption = f"{idx}. {k} ({sim*100:.1f}%)"
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

    # 5) Phase A — upload individually (always works)
    sent_file_ids: list[tuple[str, str]] = []

    try:
        for k, caption, photo_arg, should_cache in upload_tasks:
            if isinstance(photo_arg, str):
                sent_file_ids.append((k, photo_arg))
                continue

            sent = await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=photo_arg,
                caption=caption,
            )

            if sent.photo:
                fid = sent.photo[-1].file_id
                sent_file_ids.append((k, fid))
                if should_cache:
                    await db_set_card_telegram_file_id(db, k, fid)

    finally:
        for f in open_files:
            try:
                f.close()
            except Exception:
                pass

    # 6) Phase B — resend as media group (file_ids only, safe)
    # if len(sent_file_ids) >= 2:
    #    media = [InputMediaPhoto(media=fid) for _, fid in sent_file_ids]
    #    await context.bot.send_media_group(
    #        chat_id=update.effective_chat.id,
    #        media=media,
    #     )

    # 7) Send text + watch buttons
    lines = ["Top matches:"]
    keyboard: list[list[InlineKeyboardButton]] = []

    for i, (k, sim) in enumerate(candidates, start=1):
        lines.append(f"{i}. `{k}` ({sim*100:.1f}%)")
        keyboard.append(
            [InlineKeyboardButton(f"Watch {k}", callback_data=f"watch:{k}")]
        )

    await update.message.reply_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(keyboard),
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

async def post_init(application: Application):
    application.bot_data["db"] = await init_db(DB_PATH)
    await application.bot.set_my_commands([
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