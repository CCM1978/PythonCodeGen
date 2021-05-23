# -*- coding: utf-8 -*-
import socket


def client_program():
    host = socket.gethostname()  # as both code is running on same pc
    port = 5000  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server
    print("Enter your name:",end="")
    message = input(" -> ")  # take input
    name=message;
    client_socket.send(message.encode())  # send message
    print("Please Enter your Age:",end="")
    message = input(" -> ")  # take input
    Age=message
    client_socket.send(message.encode())  # send message
    print("Please Enter your Occupation:",end="")
    message = input(" -> ")  # take input
    Occupation=message
    client_socket.send(message.encode())  # send message
    if Occupation != "Doctor":
        print("\nHi there! I'm a bot here to assess your covid-19 risk.\n");
        print("Enter   'Y' for 'Yes'\n\t'N' for 'No'")
        print("Type bye if you want to exit\n")
    while ((message.lower().strip() != 'bye') and (Occupation != 'Doctor')):
        que = client_socket.recv(1024).decode()  # receive response
        print('ChatBot: ' + que,end="")  # show in terminal
        message = input(" -> ")  # again take input
        client_socket.send(message.encode())  # send message
        print("\n");
    if Occupation=="Doctor":
        print("Hey Doc, please checkout the file consisting of the detials of COVID patients.\n")
        fileData=client_socket.recv(8192).decode()
        f=open("patientsDetails.txt","w")
        f.write(fileData)
        f.close()
    client_socket.close()  # close the connection
    print("Stay safe at home and rest.\n" 
          "Wash your hands often.\n" 
          "Maintain social distancing.\n" 
          "Make sure you monitor your symptoms.\n");

if __name__ == '__main__':
    client_program()
