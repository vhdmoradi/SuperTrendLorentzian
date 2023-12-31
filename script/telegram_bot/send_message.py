import asyncio
from telegram import Bot
import os
from dotenv import load_dotenv
import datetime
import pytz
from script.utils import functions

load_dotenv()


async def send_telegram_message(
    entryexit,
    longshort,
    symbol,
    entry_price=None,
    signal_message_id=None,
    exit_price=0,
    sltp=None,
    price_diff=None,
):
    if entry_price:
        entry_price = round(entry_price, 4)
    if exit_price:
        exit_price = round(exit_price, 4)
    if price_diff:
        price_diff = round(price_diff, 4)
    bot_token = os.getenv("BOT_TOKEN")
    channel_id = os.getenv("CHANNEL_ID")
    signal_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    long_short_emoji = "🟢" if longshort == "long" else "🔴"
    entry_message = f"{long_short_emoji}{longshort} position on {symbol}.P{long_short_emoji}\n💸Entry price: {entry_price}\n🎋TP: Unknown\n❗️SL: {exit_price}$\n⏱{signal_time}{long_short_emoji}"

    exit_message = (
        f"❌Position stopped at {price_diff}% loss❌"
        if sltp == "sl"
        else f"❇️Position Targeted at {price_diff}% profit❇️"
    )
    bot = Bot(token=bot_token)
    if entryexit == "entry":
        message_object = await bot.send_message(
            chat_id=channel_id,
            text=entry_message,
        )
    elif entryexit == "exit":
        message_object = await bot.send_message(
            chat_id=channel_id,
            text=exit_message,
            reply_to_message_id=signal_message_id,
        )
    return message_object.message_id
    # try:
    #     if entryexit == "entry":
    #         message_object = await bot.send_message(
    #             chat_id=channel_id,
    #             text=entry_message,
    #         )
    #         return message_object.message_id
    #     elif entryexit == "exit":
    #         message_object = await bot.send_message(
    #             chat_id=channel_id,
    #             text=exit_message,
    #             reply_to_message_id=signal_message_id,
    #         )
    #         return message_object.message_id

    # except Exception as e:
    #     functions.log_error(e, "sending message through telegram")


loop = asyncio.get_event_loop()
