from telegram import *
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from dialog_test4 import Dialogflow

API_KEY = open('telegram_api.txt').read().strip()
updater = Updater(token=API_KEY, use_context=True)
dispatcher = updater.dispatcher
df = Dialogflow()

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hello")

def reply(update, context):
    query = update.message.text
    context.bot.send_message(chat_id=update.effective_chat.id, text=df.get_reply(query))

def caps(update, context):
    text_caps = ' '.join(context.args).upper()
    context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

def help_(update, context):
    help_text = 'Type /caps other text to archive'
    context.bot.send_message(chat_id=update.effective_chat.id, text=help_text)

dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), reply))
dispatcher.add_handler(CommandHandler('caps', caps))
dispatcher.add_handler(CommandHandler('help', help_))

print('Starting bot...')
updater.start_polling()

