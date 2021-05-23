# -*- coding: utf-8 -*-
"""
Created on Thu May 20 21:12:18 2021

@author: Vishal
"""

import random
import json
import pickle
import numpy as np

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


lemmatizer =WordNetLemmatizer()

master_data = json.loads(open("master_data.json").read())

text_word=[]
text_group=[]
text_collection=[]
skip_char=['?','!','.',',']

for temp_data in master_data['master_data']:
    for pattern in temp_data['input_key']:
        word_list = nltk.word_tokenize(pattern)
        text_word.extend(word_list)
        text_collection.append((word_list,temp_data['mark']))
        if temp_data['mark'] not in text_group:
            text_group.append(temp_data['mark'])


text_word=[lemmatizer.lemmatize(word) for word in text_word if word not in skip_char]
text_word=sorted(set(text_word))

text_group=sorted(set(text_group))
pickle.dump(text_word, open('text_word.pkl','wb'))
pickle.dump(text_group, open('text_group.pkl','wb'))

training=[]
output_empty=[0]*len(text_group)

for document in text_collection:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in text_word:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row=list(output_empty)
    output_row[text_group.index(document[1])] = 1
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
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])



trained_model = model.fit(np.array(train_x),np.array(train_y),epochs=250,batch_size=5,verbose=1)
model.save('chatbotmodel.h5',trained_model)
