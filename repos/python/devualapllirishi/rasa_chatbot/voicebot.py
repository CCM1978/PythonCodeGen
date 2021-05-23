import requests
sender = input("what is your name?\n")
bot_message = ""
while bot_message != "Bye":
    message= input("what's your message?\n")
    print("sending message now..")
    r= requests.post("http://localhost:5002/webhooks/rest/webhook",json={"sender":sender,"message":message})
    print("Bot says,",end='')
    for i in r.json():
        bot_message=i['text']
        print(f"{i['text']}")