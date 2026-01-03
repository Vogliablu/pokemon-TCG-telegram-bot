import os
import io
import logging
from dotenv import load_dotenv
from typing import Final,Iterable, List, Set, Dict, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
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
)


DB_PATH = os.getenv("SQLITE_PATH", "bot.db")

load_dotenv()
TOKEN: Final = os.environ["BOT_TOKEN"]
BOT_USERNAME: Final = '@Pokemon_Card_tracker_bot'

logger = logging.getLogger(__name__)

# Helpers

def normalize_keycode(code: str) -> str:
    return code.strip().upper()

async def classify_image(image_bytes: bytes) -> List[str]:
    """
    Replace this stub with a real HTTP call to your classifier.
    Must return a list of keycodes (strings). Example: ["AB12", "XY99"]
    """
    # TODO: implement external call (aiohttp, httpx, etc.)
    return []

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

# Private photo handler
async def handle_private_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    DM flow:
      - user sends photo to bot privately
      - bot calls classifier
      - bot replies with recognized keycodes + inline buttons to add watches
    """
    if not update.message:
        return

    try:
        image_bytes = await download_best_photo_bytes(update, context)
    except Exception as e:
        logger.exception("Failed to download DM photo: %s", e)
        await update.message.reply_text("I couldn't download that image. Please try again.")
        return

    try:
        raw_keycodes = await classify_image(image_bytes)
        keycodes = [normalize_keycode(k) for k in raw_keycodes if k and k.strip()]
        # Deduplicate while preserving order
        deduped: List[str] = []
        seen: Set[str] = set()
        for k in keycodes:
            if k not in seen:
                seen.add(k)
                deduped.append(k)
        keycodes = deduped
    except Exception as e:
        logger.exception("Classifier failed for DM photo: %s", e)
        await update.message.reply_text("The classifier failed on that image. Please try again later.")
        return

    if not keycodes:
        await update.message.reply_text(
            "No card keycode detected. Try a clearer photo (good lighting, straight angle, less glare)."
        )
        return

    keyboard = build_watch_keyboard(keycodes)

    msg_lines = [
        "Detected keycode(s):",
        *[f"- `{k}`" for k in keycodes[:12]],
    ]
    if len(keycodes) > 12:
        msg_lines.append(f"(+{len(keycodes) - 12} more not shown)")

    msg_lines.append("")
    msg_lines.append("Tap a button below to add it to your watchlist, or use /watch <keycode>.")

    await update.message.reply_text(
        "\n".join(msg_lines),
        reply_markup=keyboard,
        parse_mode="Markdown",
    )

# Group photo handler

async def handle_group_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Group flow:
      - someone posts photo in group
      - bot calls classifier
      - bot finds which users watch returned keycodes (SQLite)
      - bot notifies those users (DM by default); if DM fails, fallback to group mention
    """
    if not update.message:
        return

    chat = update.effective_chat
    msg = update.message
    if not chat:
        return

    # Optional: prevent reprocessing same message (simple in-memory dedupe)
    # Note: for true persistence you can store (chat_id, message_id) in SQLite.
    processed: Set[Tuple[int, int]] = context.application.bot_data.setdefault("processed_group_msgs", set())
    marker = (chat.id, msg.message_id)
    if marker in processed:
        return
    processed.add(marker)

    try:
        image_bytes = await download_best_photo_bytes(update, context)
    except Exception as e:
        logger.exception("Failed to download group photo: %s", e)
        return  # silently ignore in group to reduce noise

    try:
        raw_keycodes = await classify_image(image_bytes)
        keycodes = [normalize_keycode(k) for k in raw_keycodes if k and k.strip()]
        keycodes = list(dict.fromkeys(keycodes))  # dedupe preserve order
    except Exception as e:
        logger.exception("Classifier failed for group photo: %s", e)
        return

    if not keycodes:
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]
    watchers_by_code = await watchers_for_keycodes(db, keycodes)

    if not watchers_by_code:
        return

    # Build per-user list of matches
    user_hits: Dict[int, List[str]] = {}
    for code, user_ids in watchers_by_code.items():
        for uid in user_ids:
            user_hits.setdefault(uid, []).append(code)

    # Notification message: include link-ish context (group name + message id)
    group_name = chat.title or "this group"
    notif_base = f"Watched card detected in {group_name}."

    # Try DM first; if blocked / not started, fallback in group with mention
    for user_id, codes in user_hits.items():
        codes = list(dict.fromkeys(codes))
        dm_text = notif_base + "\n" + "Matched keycode(s): " + ", ".join(codes)

        dm_sent = False
        try:
            await context.bot.send_message(chat_id=user_id, text=dm_text)
            dm_sent = True
        except Forbidden:
            dm_sent = False
        except BadRequest:
            dm_sent = False
        except TelegramError as e:
            logger.warning("DM to %s failed: %s", user_id, e)
            dm_sent = False

        if not dm_sent:
            # Fallback in group: mention user_id (works even if not username)
            # Note: This may not notify if user has privacy settings; but it's a reasonable fallback.
            group_text = f"[User](tg://user?id={user_id}) watched card detected. Keycode(s): {', '.join(codes)}"
            try:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text=group_text,
                    parse_mode="Markdown",
                    reply_to_message_id=msg.message_id,
                )
            except TelegramError as e:
                logger.warning("Group fallback notify failed: %s", e)

# Callback handler for inline buttons
async def handle_watch_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles button presses from the DM identify message:
      callback_data: "watch:<KEYCODE>"
    Adds keycode to user's watchlist and acknowledges.
    """
    query = update.callback_query
    if not query or not query.data:
        return

    await query.answer()  # acknowledge quickly to stop spinner

    # Expected format: watch:AB12
    if not query.data.startswith("watch:"):
        return

    keycode = normalize_keycode(query.data.split("watch:", 1)[1])
    if not keycode:
        await query.edit_message_text("Invalid keycode.")
        return

    user = update.effective_user
    if not user:
        await query.edit_message_text("Could not identify user.")
        return

    db: aiosqlite.Connection = context.application.bot_data["db"]

    try:
        await db_add_watch(db, user.id, keycode)
    except Exception as e:
        logger.exception("Failed to add watch: %s", e)
        await query.edit_message_text("Failed to add to watchlist due to a database error.")
        return

    # Keep message, but confirm action. If you prefer not to overwrite, use query.message.reply_text instead.
    try:
        await query.edit_message_text(f"Added `{keycode}` to your watchlist.", parse_mode="Markdown")
    except TelegramError:
        # If edit fails (e.g., message too old), just send a new message.
        await context.bot.send_message(chat_id=user.id, text=f"Added {keycode} to your watchlist.")



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
    app = Application.builder().token(TOKEN).post_init(post_init).build()   
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
 



    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # Errors
    app.add_error_handler(error)


    # Run the bot
    print('Bot is running...')
    app.run_polling(poll_interval=3)