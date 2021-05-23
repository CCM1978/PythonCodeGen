# import flask dependencies
from flask import Flask, request, make_response, jsonify
import gunicorn
import ycric
import json
# initialize the flask app
app = Flask(__name__)

# default route
@app.route('/')
def index():
    return {'fulfillmentText': "first"
}

# function for responses
def results():
    # build a request object
    req = request.get_json(force=True)
    
    action = req.get('queryResult').get('action')
  
    # return a fulfillment response
    return {'fulfillmentText': 'This is a response from webhook.'}

# create a route for webhook
@app.route('/webhook', methods=['GET','POST'])
def webhook():
    data = request.get_json(silent=True)
    print(data['queryResult']['intent']['displayName'])
    if data['queryResult']['intent']['displayName']=="live" :
        return jsonify(ycric.liveMatches())
    elif data['queryResult']['intent']['displayName']=="teams": 
        #print("team")
        return jsonify(ycric.Teams(data['queryResult']['outputContexts'][0]['parameters']['number.original']))
    elif data['queryResult']['intent']['displayName']=="scoreboard": 
        return jsonify(ycric.score(data['queryResult']['outputContexts'][0]['parameters']['number.original']))
    elif data['queryResult']['intent']['displayName']=="match":
        return jsonify(ycric.liveScore(int(data['queryResult']['parameters']['number'])))
    elif data['queryResult']['intent']['displayName']=="upcomingmatch":
        return jsonify(ycric.upComing()) 

if __name__ == '__main__':
   app.run()