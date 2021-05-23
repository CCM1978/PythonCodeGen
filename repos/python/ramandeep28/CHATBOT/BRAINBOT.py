from decimal import Context
from os import write
import pyttsx3  #pip install pyttsx3.
import datetime  #pip install datetime.
import speech_recognition as sr  #pip install speechrecognition.
import wikipedia    #pip install wikipedia.
import smtplib   #import this to send mail using sendEmail function. 
import webbrowser as wb  #to implement search function.
import psutil  #pip install psutil to get cpu and battery info.
import pyjokes  #pip install pyjokes.
import os
import pyautogui  #pip install pyautogui  (to take screenshot)
import random  #for playing random song
import wolframalpha  #pip install wolframaplha==4.0.0 (error without 4.0.0!!) or pip install wolframaplha api
import json  #news
import requests  #news
from urllib.request import urlopen  #news
import time  #for stop listening query


engine = pyttsx3.init()  #pyttsx3 does text to speech conversion.
wolframalpha_app_id = 'QJ888A-XK6LLPQ38J'


# make speak function in order to execute the say command whenever we want to!!
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

speak('Hey there') #calling the speak function to execute it
print('Hey There!')

#DATE AND TIME FUNCTION
#to access the current date and time, import "datetime" library
def time_():
    Time = datetime.datetime.now().strftime("%I:%M:%S")  #we use %H for 24 hour format and use %I for 12 hour format.
    speak("The current time is:")
    speak(Time)
    print('The current time is: ',Time)


#time_()  #calling the time function to execute it

#OP will be....hey there the current time is 13 hours 14 minutes and 13 seconds!!..this format

def date_():
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    speak("Today's date is ")
    speak(day)
    speak(month)
    speak(year)
    print("Today's date is: ", day,'/',month,'/',year)

#date_()
#date and time calling are removed from here so we can call them in wishme function, in a combined and more beautiful way!!

#WISH ME FUNCTION FOR GREETING!!
def wishme():
    speak("Welcome back Ramandeep")
    print("Welcome back Ramandeep")
    date_()
    time_()

    #now gm, ga, ge and gn greetings
    hour = datetime.datetime.now().hour #we use hour to decide the time and greet according to the hour
    if hour>=4 and hour<12:
        speak("Good Morning Sir")
        print('Good Morning Sir')
    elif hour>=12 and hour<=18:
        speak("Good Afternoon Sir")
        print('Good Afternoon Sir')
    elif hour>18 and hour<=23:
        speak("Good Evening Sir")
        print('Good Evening Sir')
    else:
        speak("Good Night")
        print('Good Night')

    speak("BrainBot at your service; how can I help you today?")
    print("BrainBot at your service; how can I help you today?")

#wishme() #called wishme function to execute all the functions inside wishme
    
#TAKE COMMAND FUNCTION, if we want system to take command from us!!; for this we have to comment wishme function!!
#to let the brainbot recognise our voice, we have to INSTALL A LIBRARY and import "speechRecognition".
def TakeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:      #installed pyaudio using " pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl                      p "
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognising...")
        query = r.recognize_google(audio, language='en-US')
        print(query)

    except Exception as e:
        print(e)
        print("Please say that again...")
        return "None"
    return query

#TakeCommand()  #execute this to know try/ except. now executed in main function
def sendEmail(to, context):
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()  #ehlo helps to identify ourself to an esmtp server.
    server.starttls()  # helps start smtp in tls mode.
    # DO ENABLE LOW SECURITY IN GMAIL
    server.login('ramandeep.singh28jan@gmail.com', 'Vaheguru@12')
    server.sendmail('ramandeep.singh28jan@gmail.com',to,content)
    server.close()

def cpu():  #import psutil
    usage = str(psutil.cpu_percent())  #str to store string value; it will return cpu utilization/ usage in %.
    speak('CPU usage percent is'+usage)
    print('CPU usage percent is: '+usage)

    battery = psutil.sensors_battery()
    speak("Battery percentage is")
    speak(battery.percent)
    print('Battery Percentage is: ', battery.percent)

def joke():
    print(pyjokes.get_joke())
    speak(pyjokes.get_joke())

def screenshot():
    img = pyautogui.screenshot()
    img.save('C:/Users/lenovo/Desktop/SS.png')

   


#MAKING THE MAIN FUNCTION TO IMPLEMENT ALL FUNCTIONS MADE AND CREATING SOME NEW FUNCTIONS AS WELL FOR CHATBOT!!

if __name__ == "__main__":  # MAIN FUNCTION...

    wishme()

    while True:
        query = TakeCommand().lower()   # .lower() will store all commands in lower case for easy recognition!!

        if 'time' in query:   #will tell us time when asked
            time_()

        elif 'date' in query:   #will tell us date when asked
            date_()

        elif 'wikipedia' in query:    #TAKING RESULT FROM WIKIPEDIA
            speak("Searching...")
            query = query.replace('wikipedia', '')  #it will relace word wikipedia with a blank, so that brainbot speaks only the required answer and not unnecessary things!!!
            result = wikipedia.summary(query, sentences = 3) #it gives a result of 3 sentence.
            speak("According To Wikipedia")

            #printing and making result audible both using this
            print('According to Wikipedia: ', result)
            speak(result)

        elif 'email' in query:  #to send email to someone, import smtplib
            try:
                speak("What should I Send in Email?")
                content = TakeCommand()

                #provide reciever's email
                speak("Who's the Reciever?")
                reciever = input("Enter Reciever's email: ")
                to = reciever
                sendEmail(to, content)
                speak(content)   #to listen to what we said to send in email.
                speak("Email has been sent.")

            except Exception as e:
                print(e)    #whatever the error will be is stored in e
                speak("Unable to send email.")    #it speaks manual error given by us instead of real.

        elif 'search in chrome' in query:
            speak("What should I search?")
            chromepath = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
            #chromepath is the path where google chrome is installed.

            search = TakeCommand().lower()
            wb.get(chromepath).open_new_tab(search+'.com') #only for webites having "".com" in end

        elif 'search youtube' in query:
            speak("What should I search")
            search_term = TakeCommand().lower()
            speak("These are the results from YouTube")
            wb.open('https://www.youtube.com/results?search_query='+search_term)

        elif 'search google' in query:
            speak("What should I search?")
            search_term = TakeCommand().lower()
            speak("Here are your Google search results")
            wb.open('https://www.google.com/search?q='+search_term)

        elif 'cpu' in query:   #to know the cpu usage and battery
            cpu()

        elif 'joke' in query:   #to listen to a joke
            speak("Here's the Joke")
            joke()

        elif 'offline' in query:   #to terminate the console
            speak("Going offline Sir!!")
            speak("Hope to see you soon")
            print('Going offline sir, Hope to see you soon!!')
            quit()

        elif 'word' in query:  #to open an application
            speak("Opening MS - Word")
            ms_word = r'C:/Program Files/Microsoft Office/root/Office16/WINWORD.exe'
            os.startfile(ms_word)

        elif 'powerpoint' in query:
            speak("Opening MS-Powerpoint")
            ms_powerpoint = r'C:/Program Files/Microsoft Office/root/Office16/POWERPNT.exe'
            os.startfile(ms_powerpoint)

        elif 'excel' in query:
            speak("Opening MS-Excel")
            ms_excel = r'C:/Program Files/Microsoft Office/root/Office16/EXCEL.exe'
            os.startfile(ms_excel) 

        elif 'vs code' in query:
            speak("Opening VS-Code Studio")
            vs_code = r'C:/Users/lenovo/AppData/Local/Programs/Microsoft VS Code/Code.exe'
            os.startfile(vs_code)

        elif 'write a note' in query:    #to write a file
            speak("What to write, Sir?")
            notes = TakeCommand()   
            file = open('notes.txt', 'w')
            speak('Sir, should I include date and time?')
            ans = TakeCommand()
            if 'yes' in ans or 'sure' in ans:
                strTime = datetime.datetime.now().strftime('%I:%M:%S')
                file.write(strTime)
                file.write(' :-')
                file.write(notes)
                speak('Done taking notes, Sir')
            else:
                file.write(notes)
                speak('Done taking notes, Sir')

        elif 'show notes' in query:
            speak('Showing notes, Sir')
            file = open('notes.txt', 'r')
            print(file.read())
            speak(file.read())

        elif 'screenshot' in query:   #to take screenshot, import pyautogui
            speak("Taking screenshot, Sir")
            screenshot()
            speak("Screenshot saved to desktop")

        elif 'play music' in query:   #to play music from PC, import random
            songs_dir = 'D:/SONGS'
            music = os.listdir(songs_dir)  #stores list of song in music variable
            speak("What should I play, Sir?")
            speak("Select a number...")
            ans = TakeCommand().lower()
            while('number' not in ans and ans!= 'random' and ans!='choose yourself'):
                speak("Invalid input Sir, Please try inputting a number from 0 to 2 or say random or choose yourself")
                ans = TakeCommand().lower()
            if 'number' in ans:
                num = int(ans.replace('number', '')) #replacing number with an empty space; in order to keep it integer type.
            elif 'random' or 'choose yourself' in ans:
                num = random.randint(0,10)
            
            os.startfile(os.path.join(songs_dir,music[num])) #to join songs_dir and music and music[no] to select number from list of music

        elif 'calculate' in query:  #to do any mathematic calculation, import wolframalpha
            client = wolframalpha.Client(wolframalpha_app_id)  #it will take app_id in input
            indx = query.lower().split().index('calculate')
            query = query.split()[indx + 1:]
            res = client.query(' '.join(query))
            answer = next(res.results).text
            print('The Answer is: '+answer)
            speak('The Answer is: '+answer)

        elif 'what is' in query or 'who is' in query:  #to search meaning of anything using WOLFRAMAPLHA.
            #using the same wolframalpha id here!!
            client = wolframalpha.Client(wolframalpha_app_id)
            res = client.query(query)

            try:
                print(next(res.results).text)
                speak(next(res.results).text)
            except StopIteration:
                print("No Results")
                speak('No Results')

        elif 'remember' in query:  #to make notes and further ask the saved note
            speak('What should I remember sir?')
            memory = TakeCommand()
            speak('You asked me to remember that'+memory)
            remember = open('memory.txt', 'w')
            remember.write(memory)
            remember.close()
        elif 'do you remember anything' in query:  #part os remember, this tells the info saved by 'remember'
            remember = open('memory.txt', 'r')
            speak('You asked me to remember that: '+remember.read())

        elif 'where is' in query:  #searching any place in google maps.
            query = query.replace('where is', '') #it will replace where is with blank space
            location = query #value of query given to location i.e. the name of place given to location variable
            speak('User asked to locate'+location)
            wb.open_new_tab('https://www.google.com/maps/place/'+location)

        elif 'news' in query:  #import json, requests and (urlopen from urllib.request)
            try:     
                jsonObj = urlopen('http://newsapi.org/v2/top-headlines?sources=techcrunch&apiKey=a3622233c7fd418a8650a8d31d338095')
                data = json.load(jsonObj) #loaded data from website in data variable
                i = 1 #for (for loop) for multiple headlines.

                speak('Here are the top headlines from the Business Industry')
                print('***************TOP HEADLINES***************'+'\n')
                for item in data['articles']:
                    print(str(i)+'. '+item['title']+'\n')
                    print(item['description']+'\n')
                    speak(item['title'])
                    i+=1
                    
            except Exception as e:
                print(str(e))

        elif 'stop listening' in query:  #import time
            speak('For how many seconds you want me to stop listening, sir?')
            ans = int(TakeCommand())
            time.sleep(ans)
            print(ans)

        elif 'log out' in query:
            os.system('shutdown -l')
        elif 'restart' in query:
            os.system('shutsown /r /t 1')
        elif 'shutdown' in query:
            os.system('shutdown /s /t 1')
#pip install pyinstaller to make python file into .exe file so that other person can have access only to the brainbot app
#cd python(the directory where brainbot.py is being saved, in my case it's python file on desktop. )
#pyinstaller --onefile 'brainbot.py' (last step to create .exe file)
#the .exe file will be saved in the 'dest' named folder and will be in the same folder as that of python.
        elif 'how are you' in query:
            print('I am fine, how are you sir? I hope you are fine too')
            speak('I am fine, how are you sir? I hope you are fine too')

        elif 'who am i' in query:
            print('If you can speak, then definately you are a human!!')
            speak('If you can speak, then definately you are a human!!')

        elif 'who are you' in query:
            print('I am Brainbot, created by Raman. I can do various tasks for you. I can searh anything for you, i can play songs for you, I can search any location for you, i can tell you jokes, i can take a screenshot also, i can tell you latest news, i can save your reminders as well. You just have to command me, and i can do that for you!!')
            speak('I am Brainbot, created by Raman. I can do various tasks for you. I can searh anything for you, i can play songs for you, I can search any location for you, i can tell you jokes, i can take a screenshot also, i can tell you latest news, i can save your reminders as well. You just have to command me, and i can do that for you!!')

        elif 'why you came' in query:
            print('Thanks to Ramandeep, further it is a secret')
            speak('Thanks to Ramandeep, further it is a secret')



