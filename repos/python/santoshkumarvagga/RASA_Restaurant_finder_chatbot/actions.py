from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from smtplib import SMTPException

from rasa_sdk import Action
from rasa_sdk.events import SlotSet
import pandas as pd
import smtplib
import json

ZomatoData = pd.read_csv('zomato.csv')
ZomatoData = ZomatoData.drop_duplicates().reset_index(drop=True)
WeOperate = ['agra', 'ahmedabad', 'aligarh', 'amravati', 'amritsar', 'asansol', 'aurangabad', 'bareilly', 'belgaum',
             'bangalore', 'bhavnagar', 'bhilai', 'bhiwandi', 'bhopal', 'bhubaneswar', 'bijapur', 'bikaner', 'bilaspur',
             'bokaro steel city', 'chandigarh', 'chennai', 'coimbatore', 'cuttack', 'dehradun', 'delhi', 'dhanbad',
             'dindigul', 'durgapur', 'erode', 'faridabad', 'firozabad', 'ghaziabad', 'gorakhpur', 'gulbarga', 'guntur',
             'gurgaon', 'guwahati', 'gwalior', 'hamirpur', 'hubli–dharwad', 'hyderabad', 'indore', 'jabalpur', 'jaipur',
             'jalandhar', 'jammu', 'jamnagar', 'jamshedpur', 'jhansi', 'jodhpur', 'kakinada', 'kannur', 'kanpur',
             'karnal', 'kochi', 'kolhapur', 'kolkata', 'kollam', 'kozhikode', 'kurnool', 'lucknow', 'ludhiana',
             'madurai', 'malappuram', 'mangalore', 'mathura', 'meerut', 'moradabad', 'mumbai', 'mysore', 'nagpur',
             'nanded', 'nashik', 'nellore', 'noida', 'patna', 'puducherry', 'prayagraj', 'pune', 'purulia', 'raipur',
             'rajahmundry', 'rajkot', 'ranchi', 'ratlam', 'rourkela', 'salem', 'sangli', 'shimla', 'siliguri',
             'solapur', 'srinagar', 'surat', 'thanjavur', 'thiruvananthapuram', 'thrissur', 'tiruchirappalli',
             'tirunelveli', 'tiruvannamalai', 'ujjain', 'vadodara', 'varanasi', 'vasai-virar city',
             'vellore and warangal', 'vijayawada', 'vizag']

OurCuisine = ['american', 'chinese', 'italian', 'mexican', 'north indian', 'south indian']

valid_response = ''


def RestaurantSearch(City, Cuisine, Price_a, Price_b):
    print(f"Im action utility RestaurantSearch: {City}-{Cuisine}-{Price_a}-{Price_b}")
    City = str(City)
    Cuisine = str(Cuisine)
    Price_a = int(Price_a)
    Price_b = int(Price_b)
    if City.lower() in WeOperate and Cuisine.lower() in OurCuisine:
        TEMP = ZomatoData[(ZomatoData['Cuisines'].apply(lambda x: Cuisine.lower() in x.lower())) & (
                ZomatoData['City'].apply(lambda x: str(x).strip().lower() == City.lower())) & (
            ZomatoData['Average Cost for two'].apply(lambda x: True if Price_a <= int(x) <= Price_b else False))]
        return TEMP[['Restaurant Name', 'Address', 'Average Cost for two', 'Aggregate rating']]


class ActionCheckLocation(Action):
    def name(self):
        return 'action_check_loc'

    def run(self, dispatcher, tracker, domain):
        location = tracker.get_slot('location')
        res = False
        if location.lower() in WeOperate:
            res = True
            dispatcher.utter_message(f"=> " + "Yes, we do operate in {location}.")
        else:
            res = False
            dispatcher.utter_message("=> " + f"NO, we do not operate at {location} yet.")
        return [SlotSet('check_resp', res)]


class ActionSearchRestaurants(Action):
    def name(self):
        return 'action_search_restaurants'

    def run(self, dispatcher, tracker, domain):
        # config={ "user_key":"f4924dc9ad672ee8c4f8c84743301af5"}
        # config={ "user_key":"1sqZgyCOo36dnjmN75NSHhVHfIA_25mJd4aZ93q5VPurRC1bH"}
        loc = tracker.get_slot('location')
        cuisine = tracker.get_slot('cuisine')
        price = tracker.get_slot('price')
        print(f"Im action restaurant search func: {loc}-{cuisine}-{price}")
        if price == "Lesser than Rs. 300":
            price_a = 0
            price_b = 300
        elif price == "Rs. 300 to 700":
            price_a = 300
            price_b = 700
        elif price == "More than 700":
            price_a = 700
            price_b = 1000000
        else:
            price_a = 0
            price_b = 1000000
        results = RestaurantSearch(City=loc, Cuisine=cuisine, Price_a=price_a, Price_b=price_b)
        print(f"Results of action restaurant search: {results}")
        response = ""
        if results.shape[0] == 0:
            response = "We do not operate in that area yet"
        else:
            print("-----------------------------><")
            valid_response = RestaurantSearch(City=loc, Cuisine=cuisine, Price_a=price_a, Price_b=price_b)
            for restaurant in valid_response.iloc[:5].iterrows():
                restaurant = restaurant[1]
                response = response + f"{restaurant['Restaurant Name']} in {restaurant['Address']} has been rated " \
                                      f"{restaurant['Aggregate rating']}\n\n"
        dispatcher.utter_message("Okay DOne")
        dispatcher.utter_message("=> " + response)


class ActionSendMail(Action):
    def name(self):
        return 'action_send_mail'

    def run(self, dispatcher, tracker, domain):
        loc = tracker.get_slot('location')
        cuisine = tracker.get_slot('cuisine')
        print("im mailing action ...........")
        # results = RestaurantSearch(City=loc, Cuisine=cuisine)
        # response = ""
        # if results.shape[0] == 0:
        #     response = "We do not operate in that area yet"
        #     dispatcher.utter_message("=> " + response)
        # else:
        #     valid_response = RestaurantSearch(loc, cuisine)
        #     response = ''
        #     for restaurant in valid_response.iloc[:10].iterrows():
        #         restaurant = restaurant[1]
        #         response = response + f"{restaurant['Restaurant Name']} in {restaurant['Address']} costs " \
        #                               f"{restaurant['Average Cost for two']} on average for two and  has been rated " \
        #                               f"{restaurant['Aggregate rating']}\n\n"
        #
        # # Mail details
        # MailID = tracker.get_slot('mail_id')
        # sender = 'vaggasantoshkumar@gmail.com'
        # receivers = [str(MailID), ]
        # Message = f"This Message contains top 10 Restaurant for your query: {response} Thanks\nFoodie Bot"
        #
        # # send mail
        # try:
        #     smtpObj = smtplib.SMTP('localhost')
        #     smtpObj.sendmail(sender, receivers, Message)
        #     dispatcher.utter_message("=> " + "Emailed you Successfully, Thanks.")
        # except SMTPException:
        #     dispatcher.utter_message("=> " + "Email Cannot be Sent, Thanks.")
        # return [SlotSet('mail_id', MailID)]
        dispatcher.utter_message("Email sent")
