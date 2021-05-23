# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:43:18 2021

@author: Vishal
"""


import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer=WordNetLemmatizer()
master_data = json.loads(open("master_data.json").read())

text_word=pickle.load(open('text_word.pkl','rb'))
text_group=pickle.load(open('text_group.pkl','rb'))
model=load_model('chatbotmodel.h5')

def solve(sentence):
    bwords=bow(sentence)
    res=model.predict(np.array([bwords]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'temp_data':text_group[r[0]],'probability':str(r[1])})
        return return_list


def preprocess_input(sentence):
    line_sentence=nltk.word_tokenize(sentence)
    line_sentence=[lemmatizer.lemmatize(word) for word in line_sentence]
    return line_sentence

def bow(sentence):
    line_sentence=preprocess_input(sentence)
    bag=[0]*len(text_word)
    for w in line_sentence:
        for i,word in enumerate(text_word):
            if word==w:
                bag[i]=1
    return np.array(bag)


def interact(master_data_list,master_data_json):
    mark=master_data_list[0]['temp_data']
    list_of_master_data=master_data_json['master_data']
    for i in list_of_master_data:
        if i['mark']== mark:
            result=random.choice(i['output_response'])
            break
    return result


while True:
    with open('chatbot_input.json') as json_input:
        inp=json.load(json_input)
    message=inp["input_data"]
    ints=solve(message)
    res=interact(ints, master_data)
    dict_out={"chatbot":res}
    with open('chatbot_output.json','w') as json_output:
        json.dump(dict_out,json_output)
