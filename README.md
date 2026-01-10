# Telegram Bot for Pokemon Cards Marketplace Notifications @Pokemon_Card_tracker_bot

## Project Overview
The bot aim is to receive notifications when a **Pokemon TCG card** the user is interested in appears in a certain **telegram marketplace group**.
In telegram groups where people buy and sell Pokemon cards, it is often the case that sellers send pictures of their albums containing **multiple cards**, and checking all pictures can be **time consuming**.
With the use of this bot, is it enough to register the cards the user wants to keep an eye on, by providing a **clean picture**. The bot will then notify them once he detects that card in an image sent in the marketplace groups. 

Using the commands in the **private chat** with the bot, the user can manage their **watchlist** by **adding**, **removing** cards, show the **image of cards** in the watchlist and adjust **similarity threshold** for specific cards.
The bot checks for cards in the pictures sent in the groups it has joined. Once it detects that an image contains **one or more cards**, it computes a **similarity** for each card in the picture and each card in the user watchlist, and if any of these similarity is above a certain **threshold**, the user receives a **private message** that includes the **cropped pictures** of the cards and the **name or tag of the sender**.


## How to use
In telegram start a chat with @Pokemon_Card_tracker_bot with the command `/start`.
The general procedure is sending a picture of the card you want to receive notification for, and then adding it to the watchlist with a nickname with the command `/watch <nickname>`. 
The list of all commands can be checked using `/help`.
Other useful commands include `/watchlist` to check the watchlist, `/unwatch <nickname>` to remove a card from your watchlist, `/show <nickname>` to see an image of a card in the watchlist or `/setthreshold` to adjust the similarity threshold that needs to be met for notifications.  


## How it works
DM pipeline: photo -> cropper -> embedding -> pending prototype -> `/watch nickname` to save it

Group pipeline: photo -> cropper -> embedding -> match against watchlist -> DM notifications

The image cropping and embedding pipeline is implemented in the `vision` submodule.

## Short description of each file

- **`telegrambot.py`**  
  The file that runs the bot and should be executed to start it.  
  Defines all Telegram handlers (DM and group), command logic, and the overall orchestration of the bot.  
  It connects image processing (cropping + embeddings), database access, and user interactions.  
  This file also handles:
  - group photo monitoring and private notifications  
  - DM-only command execution  
  - in-memory caching of user prototypes and thresholds  

- **`storage.py`**  
  Database access layer.  
  Contains all SQL queries and helper functions used to read and write data from the SQLite database.  
  It abstracts operations such as:
  - creating and deleting user prototypes  
  - managing pending prototypes  
  - updating per-card thresholds  
  - retrieving data for watchlists and cache rebuilding  

- **`db.py`**  
  Database initialization and schema management.  
  Defines the SQLite schema, creates tables and indexes, and applies migrations on startup.  
  This file is responsible for:
  - setting up tables for users and cards  
  - ensuring the database is created automatically if missing  
  - handling schema evolution safely over time  

- **`vision/` (submodule)**  
  Image processing pipeline used by the bot.  
  Implements card detection and embedding extraction, including:
  - YOLO-based card croppers (user photos and group photos)  
  - ResNet-based image embedding encoder  
  - utilities for crop normalization and similarity comparison  

- **`.env` (not committed)**  
  Environment configuration (Telegram bot token, paths, thresholds).


## Requirements
- Python 3.11+ (see `requires-python` in `pyproject.toml`)
- `uv` (recommended)


## How to run the code
### 1) Clone
```bash
git clone https://github.com/Vogliablu/pokemon-TCG-telegram-bot.git
cd pokemon-TCG-telegram-bot
```
### 2) Create a Telegram bot token
Start a chat with @BotFather on telegram and obtain a bot token. Put it in a `.env` file in the project root as `BOT_TOKEN` (obviously not commited here).
### 3) Install dependencies 
Dependencies are defined in `pyproject.toml` and pinned in `uv.lock`.
```bash
 uv sync
 ```

### 4) run the bot 
```bash
uv run python telegrambot.py
```


## Privacy notes

### For users

- When you save a watched card (using `/watch <nickname>`), the bot stores:
  - a cropped image of the card,
  - its visual embedding,
  - the nickname and threshold you assigned.

- These data are stored **locally by the bot** and are kept **until you remove the card** using `/unwatch <nickname>` or `/clearwatchlist`.

- The bot does **not** store or retain:
  - full group photos,
  - group chat messages,
  - message text or captions,
  - information about other users’ watchlists.

### For group administrators

- The bot processes **photos posted in the group** in order to detect watched cards.
- The bot **never posts messages in the group**.

- When a watched card is detected, the bot sends a **private notification (DM)** to the interested user.
  - This means the bot may notify users **who are not members of the group**.
  - As a result, buyers or collectors outside the group may become aware of cards posted in the group, or the existance of the group itself.


### General notes

- The bot only processes image data necessary for card detection and matching.
- No data are shared with third parties.
- All processing happens locally on the machine where the bot is running.




  


