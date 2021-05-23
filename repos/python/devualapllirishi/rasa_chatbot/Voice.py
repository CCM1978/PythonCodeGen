import requests
import speech_recognition as sr
import os
from gtts import gTTS
import time

bot_message=""
message=""

while 1:
    r= sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything:")
        audio= r.listen(source)
        try:
            r.adjust_for_ambient_noise(audio_file)
            message= r.recognize_google(audio)
            print("You said : {}".format(message))
        except:
            print("Sorry could not recognize your voice")
    if len(message)==0:
        continue
    print("Sending message now...")
    r= requests.post('http://localhost:5002/webhooks/rest/webhook', json={"message":message})
    for i in r.json():
        bot_message=i['text']

    print(bot_message)
    myobj = gTTS(text=bot_message)
    myobj.save("welcome.mp3")
    print('saved')
    os.system("start welcome.mp3")
    if bot_message == "Bye":
        break
    time.sleep(7)
