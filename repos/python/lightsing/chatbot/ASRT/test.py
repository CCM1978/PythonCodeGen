#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于测试整个一套语音识别系统的程序
语音模型 + 语言模型
"""
from SpeechModel251 import ModelSpeech
from LanguageModel2 import ModelLanguage
from keras import backend as K

datapath = '../model/'
modelpath = '../model/'


ms = ModelSpeech(datapath)

ms.load_model(f'{modelpath}speech_model251_e_0_step_625000.model')

recognized = ms.recognize_speech_from_file('../wav/SUST11510598_010.wav')


K.clear_session()

print('*[提示] 语音识别结果：\n', recognized)
ml = ModelLanguage('../model/language')
ml.LoadModel()
recognized = ml.SpeechToText(recognized)
print('语音转文字结果：\n', recognized)














