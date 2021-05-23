# Chatbot 

# import the library
from nltk.chat.util import Chat, reflections

pairs = [
    ['(hi|heyo|Hello|halo|heu)', ['Hey there ^_^'," heyo ^_^","hilo ^_^"]],
    ["my name is (.*)",['hi %1 ^_~']],
    ['(.*) your name', ['My name is Y.U.Z.U ^_~']],
    ['(How are you|how you doing|whats up)',['oh, i am good ^_^','Cool! what about you? ^_^']],
    ['(I am good|Fine|Good)', ['Good to hear that ^_^']],
    ['(what you doing| wyd)', ['nothng much ^_^','Talking to you ^_~']],
    ['(me too|nothing)',['ohh ^_^','Nice ^_^',"Lovely ^_~ "]],
    ['(Bye|talk to you later)', ['bye ^_^','seeya ^_~', ' byeew']],
    ['(really)',['yep ^_^']]
]

chat = Chat(pairs, reflections)
chat.converse()
