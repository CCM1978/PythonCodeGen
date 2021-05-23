from __future__ import unicode_literals
from flask import Flask, render_template, request, send_file, jsonify, make_response
import os

app= Flask(__name__)


@app.route("/")
def home():    
    return render_template("main.html") 

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')    
    return str(respond(userText)) 


@app.route("/get/client_msg")
def get_custom_client_msg():
    name = request.args.get('msg')
    print(name)
    model = spacy.load('') # wwe will add an NER dir here
    test_text = name.lower()
    doc = model(test_text)
    data = '{"response":"error in parsing data"}'
    for ent in doc.ents:
        # form a proper json 
        data = {"response": ent.text}
        print(data)
        # res = make_response(jsonify(data), 200)

    return json.dumps(data) 


##TODO:
# add a setter route to store data in files/json/database
# add a module to respond() to the client request.
#   -- respond module will take bot user input , do all the API calls and stuff, parse data and return it back to this module
# using spacY and NER to get entities and form a proper response back to the user




if __name__ == "__main__":
    app.run(port=5000,debug =True)
