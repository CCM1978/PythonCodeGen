# import subprocess
# from gtts import gTTS
# mytext='Welcome to Innovative Yourself!'
# language="en"
# myobj=gTTS(text=mytext, lang=language)
# myobj.save("C:\welcome.mp3")
#
# subprocess.call(['mpg321', "welcome.mp3","--play-and-exit"])
# Import the required module for text
# to speech conversion
from gtts import gTTS

# This module is imported so that we can
# play the converted audio
import os

# The text that you want to convert to audio
mytext = 'Welcome to Medicover Hospitals!'

# Language in which you want to convert
language = 'en'

# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed
myobj = gTTS(text=mytext, lang=language,tld='co.in')
# Saving the converted audio in a mp3 file named
# welcome
myobj.save("E:/voice.mp3")

# Playing the converted file
#os.system("mpg321 welcome1.mp3")
os.system("start E:/voice.mp3")
