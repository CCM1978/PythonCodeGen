# -*- coding: utf-8 -*-
import socket
import os
from cryptography.fernet import Fernet
import encryptFunctions

# encryptFunctions.write_key()
key=encryptFunctions.load_key()
fer=Fernet(key)


file = open('questions.txt', 'r')
Questions=[line for line in file]
size=len(Questions)
def decrypter(string):
    e=string.encode()
    encrypted=e
    decrypted_encrypted=fer.decrypt(encrypted)
    #print(decrypted_encrypted.decode())
    return decrypted_encrypted.decode()


def server_program():
    # get the hostname
    host = socket.gethostname()
    port = 5000  # initiate port no above 1024
    count=0
    answers=[]
    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    server_socket.listen(50)
    conn, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))
    name= conn.recv(1024).decode()
    print("name:",name)
    Age= conn.recv(1024).decode()
    print("Age:",Age)
    Occupation= conn.recv(1024).decode()
    print("Occupation:",Occupation)
    if Occupation != "Doctor":
        while count<size:
            # receive data stream. it won't accept data packet greater than 1024 bytes
            que=conn.send(Questions[count].encode())
            ans = conn.recv(1024).decode()
            if not ans:
                # if data is not received break
                break
            if ans=="Y":
                answers.append(1)
            elif ans=="N":
                answers.append(0)
            print(name,"Q.no:",count+1,":",str(ans))
            # ans = input(' -> ')
            # conn.send(ans.encode())  # send data to the client
            count+=1
        percentage=round((sum(answers)/len(answers))*100,2);
        msg=" Dear "+name+" You have an approx "+str(percentage)+" percentage chances of having COVID19 "
        if percentage>=75:
            msg+="\nYou are having a high risk of getting COVID19\nPlease visit a nearby hospital and take a COVID test\nOr you may contact helpline "
        elif percentage<75 and 50<=percentage:
            msg+="\nYou are having an average risk of gettign COVID19\nSo please make sure that you are quarantined for 2 weeks minimum"
        else:
            msg+="\nYou are having a low risk of having COVID19\nDon't worry much\nTake Care."
        conn.send(msg.encode())
        conn.recv(1024).decode()
        path = 'D:\Project\chatbot\clients'
        f=open(os.path.join(path, name+","+Age+".txt"), 'a')
        s="Patient name:"+name+"\n"
        for i in range(len(answers)):
            if answers[i]==1:
                a="|Yes"
            else:
                a="|No"
            s+=Questions[i]+"\n"+a+"\n"
        s=s.encode()
        s=fer.encrypt(s)
        f.write(s.decode())
        f.close()
    if Occupation=="Doctor":
        path = 'D:\Project\chatbot\clients'
        out=""
        out+="Questions\n"
        for i in range(len(Questions)):
            out+=Questions[i]+"\n"
        filenames = [i for i in os.listdir(path)]
        for fname in filenames:
            with open(os.path.join(path, fname)) as infile:
                count=0
                out+="\n"+"Patient Name:"+fname[:-4]+"\n"
                filedata=infile.read()
                decrypted_encrypted=decrypter(filedata)
                for line in decrypted_encrypted:
                    if line[0]=="Y" or line[0]=="N":
                        count+=1
                        out+="Q.no"+str(count)+": "+line+" "
        conn.send(out.encode())
    conn.close()  # close the connection


if __name__ == '__main__':
    server_program()

