"""Chatbot with Algolia Answers sample application in Python"""
import os

import requests
from dotenv import load_dotenv, find_dotenv
from flask import Flask, render_template, jsonify, request

load_dotenv(find_dotenv())

# Setup Flask
STATIC_DIR = str(
    os.path.abspath(os.path.join(
        __file__,
        '..',
        os.getenv('STATIC_DIR')
    ))
)
app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="",
    template_folder=STATIC_DIR
)

@app.route('/', methods=['GET'])
def home():
    """Display the chatbot interface."""
    return render_template('index.html')

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle the Dialogflow webhook"""

    # We get the query from the Dialogflow request.
    # https://cloud.google.com/dialogflow/es/docs/fulfillment-webhook#webhook_request
    query = request.json
    query_text = query['queryResult']['queryText']
    
    # Query the Algolia Answers API.
    # https://www.algolia.com/doc/guides/algolia-ai/answers/#finding-answers-in-your-index
    headers = {
    'Content-Type': 'application/json',
    'X-Algolia-Api-Key': os.getenv("ALGOLIA_API_KEY"),
    'X-Algolia-Application-ID': os.getenv("ALGOLIA_APP_ID"),
    }
    url = f'https://{os.getenv("ALGOLIA_APP_ID")}-dsn.algolia.net/1/answers/{os.getenv("ALGOLIA_INDEX_NAME")}/prediction'
    data = {
        "query": query_text,
        "queryLanguages": ["en"],
        "attributesForPrediction": ["a"],
        "nbHits": 1
    }
    results = requests.post(url, headers=headers, json=data).json()

    try:
        hit = results['hits'][0]
        # https://cloud.google.com/dialogflow/es/docs/fulfillment-webhook#webhook_response
        webhook_response = {
            'fulfillment_messages': [
                {
                    'payload': {
                        'richContent': [
                            [
                                {
                                    "type": "description",
                                    "text": [
                                        f'...{hit["_answer"]["extract"]}...'.replace('<em>', '').replace('</em>', '')
                                    ]
                                }
                            ]
                        ]
                    }
                }
            ]
        }
        return jsonify(webhook_response), 200
    except IndexError:
        # If we didn't get any hits, we return some question suggestions.
        webhook_response = {
            'fulfillment_messages': [
                {
                    'text': {
                        'text': [
                            'Sorry, I do not have an answer to that!',
                            'Here are some suggestion of questions:'
                        ]
                    }
                },
                {
                    'payload': {
                        'richContent': [
                            [
                                {
                                    "type": "chips",
                                    "options": [
                                        {
                                            "text": "Are the COVID-19 Vaccines safe?"
                                        },
                                        {
                                            "text": "Can my COVID-19 test come back positive if I get vaccinated?"
                                        }
                                    ]
                                }
                            ]
                        ]
                    }
                }
            ]
        }
        return jsonify(webhook_response), 200
