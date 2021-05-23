# Put your chatbot code when you made a text chatbot here


from chatterbot import ChatBot
# Giving the ChatterBot a name
chatbot = ChatBot('Ron Obvious')
# importing the corpus
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer


conversation = [
    "Hello",
    "Hi there!",
    "How are you doing?",
    "I'm doing great.",
    "That is good to hear",
    "Thank you.",
    "You're welcome."
]



print("im at the start?")

trainer = ListTrainer(chatbot)

trainer.train(conversation)


#the trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)
#telling the chattbot for language
trainer.train("chatterbot.corpus.english")

print("welcome")


    


''' ******************* GUI Below Engine Above **************** '''
# Import for the GUI 
from chatbot_gui import ChatbotGUI
# create the chatbot app
app = ChatbotGUI("The stick dude")
# time
import time


# define the function that handles incoming user messages
@app.event
def on_message(chat: ChatbotGUI, text: str):
    # this is where you can add chat bot functionality!
    # text is the text the user has entered into the chat
    # you can use chat.send_ai_message("") to send a message as the AI
    # you can use chat.start_gif() and chat.stop_gif() to start and stop the gif
    # you can use chat.clear() to clear the user and AI chat boxes

    # print the text the user entered to console
    print("User Entered Message: " + text)
    # chat.start_gif()
    # send a response as the AI
    #chat.send_ai_message("Hello!")
    bot_response = chatbot.get_response(text)
    
    chat.send_ai_message(bot_response)
    

    # if the user send the "clear" message clear the chats
    if text == "clear":
        chat.clear()
    if text.lower().find("bye") != -1:
        # loging out
        print("Good bye")
        


# run the chat bot application
app.run()




# is_exit = False

# # opt_1 = input("Do you what the AI to talk to its self? (y/n) " )

# while is_exit == False:
#     #get user input:
#     user_input = input()
#     #Intercept specic user respance override chat bot.
#     #we will do this by testing a consition first
#     if user_input.lower().find("bye") != -1:
#         #loging out
#         is_exit = True
#         print("Good bye")
