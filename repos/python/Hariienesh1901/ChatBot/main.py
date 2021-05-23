import pywhatkit as kit
import time

print('1. Play YouTube Video or Songs.')
print('2. Search in Google.')
print('3. Get info about anything.')
print('4. Text to handwriting.')
print('5. Send a Email.')

UserRequest = input('Enter 1, 2 ,3, 4 or 5: ')

if UserRequest == '1':
    Video_Song = input('What song or video do you want to play: ')
    kit.playonyt(Video_Song)

if UserRequest == '2':
    Search = input('What do you want to search: ')
    kit.search(Search)

if UserRequest == '3':
    About = input('What do you want to know about: ')
    kit.info(About)
    time.sleep(60)

if UserRequest == '4':
    Text = input('Please enter the text that you want to make as a handwriting: ')
    kit.text_to_handwriting(Text)

if UserRequest == '5':
    Email = input('Your Email: ')
    Password = input('Your Password: ')
    Title = input('Email title: ')
    Body = input('Your Email body: ')
    To = input('Email of the person you want to send this Email: ')
    kit.send_mail(Email, Password, Title, Body, To)
