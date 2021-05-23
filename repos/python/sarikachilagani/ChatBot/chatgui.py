import tkinter
from tkinter import *

from chatbot import chatbot_response


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("chatbot")
base.geometry("500x500")
base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',command= send )

EntryBox = Text(base, bd=0, bg="white",width="45", height="5", font="Arial")

scrollbar.place(x=480,y=8, height=386)
ChatLog.place(x=6,y=6, height=386, width=470)
EntryBox.place(x=128, y=401, height=90, width=350)
SendButton.place(x=6, y=401, height=90)
base.mainloop()