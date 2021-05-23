from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS


import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


app = Flask(__name__)
CORS(app)
api = Api(app)

lematizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words=[]
classes=[]
documents=[]
ignore_letters=['?','!','.',',']

nltk.download('punkt')
for intent in intents['intents']:
  for pattern in intent['patterns']:
    word_list = nltk.word_tokenize(pattern)
    words.extend(word_list)
    documents.append((word_list, intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])

print(documents[0][1])

nltk.download('wordnet')
words=[lematizer.lemmatize(word) for word in words if word not in ignore_letters]
words=sorted(set(words))
classes=sorted(set(classes))

print(words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training=[]
output_empty=[0]*len(classes)
for document in documents:
  bag=[]
  word_patterns=document[0]
  word_patterns=[lematizer.lemmatize(word.lower()) for word in word_patterns]
  for word in words:
    bag.append(1) if word in word_patterns else bag.append(0)

  output_row=list(output_empty)
  output_row[classes.index(document[1])]=1
  training.append([bag,output_row])

random.shuffle(training)
training=np.array(training)

train_x=list(training[:,0])
train_y=list(training[:,1])

model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd=SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbotmodel.h5',hist)
print("Done")

#testing the model

from tensorflow.keras.models import load_model

lematizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))

model=load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
  sentence_words=nltk.word_tokenize(sentence)
  sentence_words=[lematizer.lemmatize(word) for word in sentence_words]
  return sentence_words

def bag_of_words(sentence):
  sentence_words=clean_up_sentence(sentence)
  bag=[0]*len(words)
  for w in sentence_words:
    for i,word in enumerate(words):
      if word==w:
        bag[i]=1
  return np.array(bag)

def predict_class(sentence):
  bow=bag_of_words(sentence)
  res=model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD=0.25
  results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
  results.sort(key=lambda x : x[1], reverse=True)
  return_list=[]
  for r in results:
    return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
  return return_list

def get_response(intents_list,intents_json):
  tag=intents_list[0]['intent']
  list_of_intents=intents_json['intents']
  for i in list_of_intents:
    if i['tag']==tag:
      result=random.choice(i['responses'])
      break
  return result

print("Go! Bot is running!!")

def call_query(key):
    ints=predict_class(key)
    res=get_response(ints,intents)
    # if res in ["Sad to see you go:(","Talk to you later","Good Bye!"]:
    #   break
    return res




    

    

    
    # jsonString=json.dumps(str(train_predict))+employee

    # return jsonString





class chat(Resource):

    def get(self):
        query = request.args.get('query')

        out= call_query(query)

        return out

api.add_resource(chat, '/')

if __name__ == '__main__':
    app.run(debug=True)
