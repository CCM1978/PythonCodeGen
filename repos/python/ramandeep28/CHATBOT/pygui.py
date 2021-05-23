import tkinter
from tkinter import messagebox #this pops up a display, showing us the message (Here used in the function of #4, button widget)
root=tkinter.Tk()
#1
root.title("Raman Deep Singh")
#2
root.geometry("500x500") #it increases the size of the outpot window.
#3
label=tkinter.Label(root, text="Which one is your favourite?")
label.pack()
#4
def buttonTapped():
    messagebox._show("Message(TITLE)", "Button Clicked(BODY)")

button=tkinter.Button(root, text="Click Me", command=buttonTapped).pack() #command ke name ka function bnayege ji

#5
check1=tkinter.Checkbutton(root, text="Apple").pack()
check2=tkinter.Checkbutton(root, text="Mango").pack()
check3=tkinter.Checkbutton(root, text="Banana").pack()

#6
entry=tkinter.Entry(root, bg='white', fg= 'red')
entry.pack()

def cleartxt():
    entry.delete(0, 9999)
tkinter.Button(root, text= 'Clear Me', command= cleartxt).pack()

def showtxt():
    tkinter.Label(text= "The entered text is: {}".format(entry.get())).pack()
tkinter.Button(root, text= 'Show text on label', command= showtxt).pack()

#7



root.mainloop() #its a continuous loop which keeps the O/P window on screen uptil we press x button ji.


#ithe se window bnn gayi, ab WIDGETS ADD krenge g.
#1. title widget
#2. geometry widget
#3. Label widget
#4. Button Widget
#5. CheckButton widget
#6. Entry widget
#7. Radiobutton widget