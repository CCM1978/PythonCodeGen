import mysql.connector
from actions import result


def DataUpdate(name: object, age: object, gender: object, location: object, contact: object) -> object:
    mydb = mysql.connector.connect(host="localhost", user="root", passwd="", database="anusha")
    mycursor = mydb.cursor()
    sql = 'INSERT INTO details (name,age,gender,location,contact) VALUES("{0}","{1}","{2}","{3}","{4}");'.format(name, age, gender, location, contact)
    # val = ("", "", "")
    mycursor.execute(sql, result)
    mydb.commit()
    print(mycursor.rowcount, "record inserted")
