from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

bot = ChatBot('chatbot')
trainer = ChatterBotCorpusTrainer(bot)
trainer.train('chatterbot.corpus.chinese')
