from flask import Flask,request,jsonify
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
import urllib.request
import time

import os
import sys

app=Flask(__name__)


point = []
carbon = []
path='chromedrver.exe'
driver=webdriver.Chrome(path)


@app.route('/walk',methods=['POST']) #json으로 들어온 사용자 요청을 보고 판단
def walk():

    req = request.get_json()
    params = req['action']['detailParams']
    res={
        "version" : "2.0",
        "template" : {
            "outputs":[
                {
                    "simpleText":{
                        "text":"걷기 시작한 위치와 끝낸 위치를 하나씩 차례대로 입력하세요 \n 예시) 혜화역 \n    서울역"
                         }
                     }
                 ]
              }
          }

    return jsonify(res)
        
    if 'sys_location' not in params.keys():
        res={
            "version" : "2.0",
            "template" : {
                "outputs":[
                    {
                        "simpleText":{
                            "text":"걷기 시작한 위치와 끝낸 위치를 하나씩 차례대로 입력하세요"
                            }
                        }
                    ]
                }
            }

        return jsonify(res)
        
    if 'sys_location' in params.keys():
        location = params['sys_location']['value']
    if 'sys_location1' in params.keys():
        location1 = params['sys_location1']['value']
    

    location_encoding = urllib.parse.quote(location+'에서'+location1+'까지')
    url='https://search.naver.com/search.naver?sm=top_hty&fbm=1&ie=utf8&query='%(location_encoding)

    req=Request(url)
    page=urlopen(req)
    
    #options=webdriver.ChromeOptions()
    #options.headless=True
    #driver = webdriver.Chrome('./chromedriver',options=options)  
    
    walk_check=driver.find_element_by_css_selector('input#raa4').click()
    time.sleep(1)
    walk_check2=driver.find_element_by_css_selector('button._ft_search').click()
    time.sleep(1)
        
    walk_km=driver.find_element_by_xpath('//*[@id="_fasttrack"]/div[2]/div/div[7]/div/strong[2]/em').text 
    walk_unit1=driver.find_element_by_xpath('//*[@id="_fasttrack"]/div[2]/div/div[7]/div/strong[2]').text.replace(walk_km,'')
    if walk_unit1 == 'm':
        walk_km_real=float(walk_km) * 0.001
        answer = '%f km 걸으셨네요! \n\n'%(walk_km_real)
        walk_carbon = 140 * walk_km_real
        walk_point = walk_carbon * 0.05
        carbon.append(walk_carbon)
        point.append(walk_point)
        answer += '약 %f g의 탄소 배출을 줄이셨어요! \n %f 탄소 포인트가 적립되었습니다.'%(walk_carbon,walk_point)
        #get_point(walk_carbon)
        #get_carbon(walk_carbon)
            
    elif walk_unit1 == 'km':
        answer = '%f km 걸으셨네요! \n\n'%(walk_km_real)
        walk_carbon = 140 * float(walk_km)
        walk_point = walk_carbon * 0.05
        #carbon_amount += walk_carbon
        #print(f'약 {walk_carbon}g의 탄소 배출을 줄이셨어요! \n{walk_point} 탄소 포인트가 적립되었습니다.')
        #get_point(walk_carbon)
        #get_carbon(walk_carbon)
        carbon.append(walk_carbon)
        point.append(walk_point)
        answer += '약 %f g의 탄소 배출을 줄이셨어요! \n %f 탄소 포인트가 적립되었습니다.'%(walk_carbon,walk_point)


    res = {
        "version":"2.0",
        "template": {
            "outputs":[
                {
                    "simpleText":{
                        "text": answer
                        }
                    }
                ]
            }
        }

    return jsonify(res)


@app.route('/bike',methods=['POST']) #json으로 들어온 사용자 요청을 보고 판단
def bike():
    #content = request.get_json()
    #content = content['userRequest']
    #content = content['utterance'] #사용자가 전송한 실제 메시지

    req = request.get_json()
    params = req['action']['detailParams']
    res={
        "version" : "2.0",
        "template" : {
            "outputs":[
                {
                    "simpleText":{
                        "text":"자전거 타기 시작한 위치와 끝낸 위치를 하나씩 차례대로 입력하세요 \n 예시) 혜화역 \n    서울역"
                         }
                     }
                 ]
              }
          }

    return jsonify(res)
        
    if 'sys_location' not in params.keys():
        res={
            "version" : "2.0",
            "template" : {
                "outputs":[
                    {
                        "simpleText":{
                            "text":"자전거 타기 시작한 위치와 끝낸 위치를 하나씩 차례대로 입력하세요"
                            }
                        }
                    ]
                }
            }

        return jsonify(res)
        
    if 'sys_location' in params.keys():
        location = params['sys_location']['value']
    if 'sys_location1' in params.keys():
        location1 = params['sys_location1']['value']
    

    location_encoding = urllib.parse.quote(location+'에서'+location1+'까지')
    url='https://search.naver.com/search.naver?sm=top_hty&fbm=1&ie=utf8&query='%(location_encoding)

    req=Request(url)
    page=urlopen(req)
    
    #driver = webdriver.Chrome('./chromedriver',options=options)  
    #driver.get(urlnaver)
        
    #walk_point = [] #걷기 포인트
    #walk_carbon = [] #걷기 탄소량
    
    bike_check=driver.find_element_by_css_selector('input#raa3').click()
    time.sleep(1)
    bike_check2=driver.find_element_by_css_selector('button._ft_search').click()
    time.sleep(1)
        
    bike_km=driver.find_element_by_xpath('//*[@id="_fasttrack"]/div[2]/div/div[7]/div/strong[2]/em').text 
    bike_unit1=driver.find_element_by_xpath('//*[@id="_fasttrack"]/div[2]/div/div[7]/div/strong[2]').text.replace(bike_km,'')
    if bike_unit1 == 'm':
        bike_km_real=float(bike_km) * 0.001
        answer = '%f km 자전거 타셨네요! \n\n'%(bike_km_real)
        bike_carbon = 140 * bike_km_real
        bike_point = bike_carbon * 0.05
        carbon.append(bike_carbon)
        point.append(bike_point)
        answer += '약 %f g의 탄소 배출을 줄이셨어요! \n %f 탄소 포인트가 적립되었습니다.'%(bike_carbon,bike_point)
       
            
    elif bike_unit1 == 'km':
        answer = '%f km 자전거 타셨네요! \n\n'%(bike_km_real)
        bike_carbon = 140 * float(bike_km)
        bike_point = bike_carbon * 0.05
        #carbon_amount += walk_carbon
        #print(f'약 {walk_carbon}g의 탄소 배출을 줄이셨어요! \n{walk_point} 탄소 포인트가 적립되었습니다.')
        #get_point(walk_carbon)
        #get_carbon(walk_carbon)
        carbon.append(bike_carbon)
        point.append(bike_point)
        answer += '약 %f g의 탄소 배출을 줄이셨어요! \n %f 탄소 포인트가 적립되었습니다.'%(bike_carbon,bike_point)


    res = {
        "version":"2.0",
        "template": {
            "outputs":[
                {
                    "simpleText":{
                        "text": answer
                        }
                    }
                ]
            }
        }

    return jsonify(res)
    

@app.route('/courage',methods=['POST']) #json으로 들어온 사용자 요청을 보고 판단
def courage():
    content = request.get_json()
    content = content['userRequest']
    content = content['utterance'] #사용자가 전송한 실제 메시지

    res={
        "version" : "2.0",
        "template" : {
            "outputs":[
                {
                    "simpleText":{
                        "text":"용기내 챌린지에 참여한 횟수를 입력하세요 \n숫자만 입력하세요"
                         }
                     }
                 ]
              }
          }

    return jsonify(res)

    courage_carbon= 227 * int(content)
    courage_point= 0.05 * courage_carbon
    carbon.append(courage_carbon)
    point.append(courage_point)

    answer = '약 %f g의 탄소 배출을 줄이셨어요! \n %f 탄소 포인트가 적립되었습니다.'%(courage_carbon,courage_point)

    res = {
        "version":"2.0",
        "template": {
            "outputs":[
                {
                    "simpleText":{
                        "text": answer
                        }
                    }
                ]
            }
        }

    return jsonify(res)


@app.route('/tumbler',methods=['POST']) #json으로 들어온 사용자 요청을 보고 판단
def tumbler():
    content = request.get_json()
    content = content['userRequest']
    content = content['utterance'] #사용자가 전송한 실제 메시지

    res={
        "version" : "2.0",
        "template" : {
            "outputs":[
                {
                    "simpleText":{
                        "text":"텀블러를 사용한 횟수를 입력하세요 \n숫자만 입력하세요"
                         }
                     }
                 ]
              }
          }

    return jsonify(res)

    tumbler_carbon= 11 * int(content)
    tumbler_point= 0.05 * tumbler_carbon
    carbon.append(tumbler_carbon)
    point.append(tumbler_point)

    answer = '약 %f g의 탄소 배출을 줄이셨어요! \n %f 탄소 포인트가 적립되었습니다.'%(tumbler_carbon,tumbler_point)

    res = {
        "version":"2.0",
        "template": {
            "outputs":[
                {
                    "simpleText":{
                        "text": answer
                        }
                    }
                ]
            }
        }

    return jsonify(res)
   
@app.route('/mail',methods=['POST']) #json으로 들어온 사용자 요청을 보고 판단
def mail():
    content = request.get_json()
    content = content['userRequest']
    content = content['utterance'] #사용자가 전송한 실제 메시지

    res={
        "version" : "2.0",
        "template" : {
            "outputs":[
                {
                    "simpleText":{
                        "text":"지운 이메일의 개수를 입력하세요 \n숫자만 입력하세요"
                         }
                     }
                 ]
              }
          }

    return jsonify(res)

    mail_carbon= 4 * int(content)
    mail_point= 0.05 * mail_carbon
    carbon.append(mail_carbon)
    point.append(mail_point)

    answer = '약 %f g의 탄소 배출을 줄이셨어요! \n %f 탄소 포인트가 적립되었습니다.'%(mail_carbon,mail_point)

    res = {
        "version":"2.0",
        "template": {
            "outputs":[
                {
                    "simpleText":{
                        "text": answer
                        }
                    }
                ]
            }
        }

    return jsonify(res)

@app.route('/carbonamount',methods=['POST'])
def carbonamount():

    carbon_amount=sum(carbon)
    answer = '지금까지 줄인 탄소는 총 %f g 입니다. \n\n'%(carbon_amount)
    tree=carbon_amount//6600
    answer += '30년생 소나무 %d 그루가 1년동안 흡수하는 이산화탄소량입니다!'%(tree)
    res = {
        "version":"2.0",
        "template": {
            "outputs":[
                {
                    "simpleText":{
                        "text": answer
                        }
                    }
                ]
            }
        }

    return jsonify(res)

@app.route('/pointamount',methods=['POST'])
def pointamount():

    point_amount=sum(point)
    answer = '지금까지 누적된 탄소 포인트는 %f점 입니다. \n\n'%(point_amount)
    point_tree=point_amount//300
    answer += '현재 %d그루의 나무를 심을 수 있습니다!'%(point_tree)
    res = {
        "version":"2.0",
        "template": {
            "outputs":[
                {
                    "simpleText":{
                        "text": answer
                        }
                    }
                ]
            }
        }

    return jsonify(res)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True,threaded=True)
