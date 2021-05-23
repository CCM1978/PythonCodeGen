from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from rasa_sdk import Action
from rasa_sdk.events import SlotSet
import pandas as pd
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

ZomatoData = pd.read_csv('zomato.csv')
ZomatoData = ZomatoData.drop_duplicates().reset_index(drop=True)
WeOperate = ['Delhi', 'Gurgaon', 'Noida', 'Faridabad', 'Allahabad', 'Bhubaneshwar', 'Mangalore', 'Mumbai', 'Ranchi', 'Patna', 'Mysore', 'Aurangabad', 'Amritsar', 'Puducherry', 'Varanasi', 'Nagpur', 'Vadodara', 'Dehradun', 'Vizag', 'Agra',
             'Ludhiana', 'Kanpur', 'Lucknow', 'Surat', 'Kochi', 'Indore', 'Ahmedabad', 'Coimbatore', 'Chennai', 'Guwahati', 'Jaipur', 'Hyderabad', 'Bangalore', 'Nashik', 'Pune', 'Kolkata', 'Bhopal', 'Goa', 'Chandigarh', 'Ghaziabad', 'Ooty', 'Gangtok', 'Shimla']
Cuisines = ['Chinese', 'Mexican', 'Italian',
            'American', 'South Indian', 'North Indian']


class ActionSearchRestaurants(Action):
    def __init__(self):
        self.operate_city = [city.lower() for city in WeOperate]
        self.cuisine = [cuisine.lower() for cuisine in Cuisines]

    def name(self):
        return 'action_search_restaurants'

    def run(self, dispatcher, tracker, domain):
        loc = tracker.get_slot('location')
        cuisine = tracker.get_slot('cuisine')
        budget = tracker.get_slot('budget')

        response = ""

        if(self.validate_city(loc) == False):
            response = F"We do not operate in {loc}"
            dispatcher.utter_message(response)
            return []

        if(self.validate_cuisine(cuisine) == False):
            response = F"We do not serve {cuisine} food"
            dispatcher.utter_message(response)
            return []

        results = self.restaurant_search(loc, cuisine, budget)

        if results.shape[0] == 0:
            response = "no results"
            dispatcher.utter_message(response)
            return []
        else:
            response = 'Top 5 search results: \n\n'
            response = response + '-' * 50
            for restaurant in results.iloc[:5].iterrows():
                restaurant = restaurant[1]
                response = response + \
                    F"{restaurant['Restaurant Name']} in {restaurant['Address']} rated {restaurant['Aggregate rating']} \n\n"

        dispatcher.utter_message(response)

        # search result for email
        search_result = ""
        for restaurant in results.iloc[:10].iterrows():
                restaurant = restaurant[1]
                search_result = search_result + \
                    F"{restaurant['Restaurant Name']} in {restaurant['Address']} rated {restaurant['Aggregate rating']} with avg cost {restaurant['Average Cost for two']} \n\n"

        return [SlotSet('location', loc), SlotSet('response', search_result)]

    def validate_city(self, location):
        return location.lower() in self.operate_city

    def validate_cuisine(self, cuisine):
        return cuisine.lower() in self.cuisine

    def isinbudget(self, value, budget_type):
        if(budget_type == 'Lesser than 300'):
            return value <= 300

        if(budget_type == 'Rs. 300 to 700'):
            return value > 300 and value <= 700

        if(budget_type == 'More than 700'):
            return value > 700

    def restaurant_search(self, city, cuisine, budget_type):
        TEMP = ZomatoData[(ZomatoData['Cuisines'].apply(lambda x: cuisine.lower() in x.lower())) & (
            ZomatoData['City'].apply(lambda x: city.lower() in x.lower())) & (
                ZomatoData['Average Cost for two'].apply(
                    lambda x: self.isinbudget(x, budget_type))
        )]

        # sort by rating
        TEMP.sort_values(by='Aggregate rating', ascending=False, inplace=True)
        return TEMP[['Restaurant Name', 'Address', 'Average Cost for two', 'Aggregate rating']]


class ActionSendMail(Action):
    def name(self):
        return 'action_send_mail'

    def run(self, dispatcher, tracker, domain):
        mailID = tracker.get_slot('email')
        response = tracker.get_slot('response')

        self.sendmail(mailID, response)
        return [SlotSet('email', mailID)]

    def sendmail(self, mailID, response):
        mail_content = F"Hello,\n\nGet ready to eat! Your search results are here:\n\n{response}"
        if(mailID != None):
            return
        # The mail addresses and password
        sender_address = 'sender123@gmail.com'
        sender_pass = 'xxxxxxxx'
        receiver_address = mailID
        # Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        # The subject line
        message['Subject'] = 'Your restaurant search result is ready!'
        # The body and the attachments for the mail
        message.attach(MIMEText(mail_content, 'plain'))
        # Create SMTP session for sending the mail
        session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
        session.starttls()  # enable security
        # login with mail_id and password
        session.login(sender_address, sender_pass)
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()
