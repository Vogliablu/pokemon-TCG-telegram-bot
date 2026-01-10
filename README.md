# Telegram Bot for Pokemon Cards Marketplace Notifications @Pokemon_Card_tracker_bot

## Project Overview
The bot aim is to receive notifications when a Pokemon TCG card the user is interested in appears in a certain telegram marketplace group.
In telegram groups where people buy and sell Pokemon cards, it is often the case that sellers send pictures of their albums containing multiple cards, and checking all pictures can be time consuming.
With the use of this bot, is it enough to register the cards the user wants to keep an eye on, by providing a clean picture. The bot will then notify them once he detects that card in an image sent in the marketplace groups. 

Using the commands in the private chat with the bot, the user can manage their watchlist by adding, removing cards, show the image of cards in the watchlist and adjust similarity threshold for specific cards.
The bot checks for cards in the pictures sent in the groups it has joined. Once it detects that an image contains one or more cards, it computes a similarity for each card in the picture and each card in the user watchlist, and if any of these similarity is above a certain threshold, the user receives a private message that includes the cropped pictures of the cards and the name or tag of the sender.

## How to use
In telegram start a chat with @Pokemon_Card_tracker_bot with the command `/start`.
The general procedure is sending a picture of the card you want to receive notification for, and then adding it to the watchlist with a nickname with the command `/watch <nickname>`. 
The list of all commands can be checked using `/help`.
Other useful commands include `/watchlist` to check the watchlist, `/unwatch <nickname>` to remove a card from your watchlist, `/show <nickname>` to see an image of a card in the watchlist or `/setthreshold` to adjust the similarity threshold that need be met for notifications.  

