from flask import Flask, render_template, request, url_for
from train_chatbot import getResponse

app = Flask(__name__)


@app.route('/reqdata', methods=["GET", "POST"])
def reqdata():
    if (request.get_data("req") != None) and (request.method == "POST"):
        data = request.get_data("req").decode()
        response = getResponse(data)
        return response


@app.route('/')
def main():
    return render_template('bot.html')


if __name__ == "__main__":
    app.run()
