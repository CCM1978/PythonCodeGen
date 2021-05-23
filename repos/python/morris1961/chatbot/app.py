# -*- coding:utf-8 -*-

import os
from datetime import datetime

from flask import Flask, abort, request

# https://github.com/line/line-bot-sdk-python
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    ButtonsTemplate,
    MessageAction,
    TemplateSendMessage,
)

QA = {
    "自我介紹": "我是王梓旭，目前就讀國立台灣大學資訊管理學系三年級。首先，我喜歡與人合作、甚至是擔任領導者，高中時，曾經擔任過社團社長，舉辦過許多多校合辦的大型活動，而我在活動中分別擔任過總召以及副召，我也因此培養了領導能力，也學到了如何與團隊其他成員共事。再來，我也喜歡幫助他人，從國中一直到現在，許多同儕都會向我詢問課業上或是其他問題，我也都會在我的能力範圍內，盡力的去協助他們解決問題。同時，我目前也正擔任程式教學服務學習團隊的助教，透過去地區國中進行 python 程式教學，我因此了解到還有很多人需要幫助。最後，我也很喜歡學習程式的相關知識，並將學習到的知識活用在不同地方，像是我做過的數次課程相關專案，包含用 c++ 做最佳化問題，成果的品質為小組競賽中的前5%、開發 RPG 遊戲、用 unity 製作遊戲、用 python 做統計以及作業研究相關專案，例如：提供拖吊場的路線進行最佳化。除此之外，我也有良好的網路技術基礎，我目前正在修習nodejs的課程，以及如何擔任網管的課，對整個網路概念相當熟稔。在課餘時間，我也和教授參與了一項產學合作專案，替廠商設計生產排程規劃的軟體服務，針對使用者需求做規劃與調整，這次經驗使我與客戶溝通的能力提升，也更了解商務上軟體的實際運作。",
    "動機": "line是有名的大公司，如果能夠進入公司內實習，必定對我的未來有很大的幫助，不管是體會職場的一些甘苦，或是完成一些實務面的專案，這一切都是在學校內無法體會到的，因此我想要學習如何面對這樣的專案。",
    "優點": "願意且喜歡幫助他人，就跟我前面自我介紹時提到的一樣，這樣的個性幫助我不管在哪裡都可以迅速融入同儕，我每次到新學校或是新環境時，通常都是因此而跟大家成為朋友。除此之外，我也不吝於主動向同儕們請教我不懂的地方，導致互利互惠的結果。",
    "缺點": "做事情容易衝動，曾經因為意見不合與朋友發生爭吵，但卻不願各退一步。現在我也正在努力改變自己，讓自己習慣做事情都要三思而後行，以避免發生不可挽回的憾事。",
    "想要帶走什麼": "希望可以學到學校裡不會教的，較實務面的經驗，了解其中差距，並改進，因此而成長，例如開發多人大型專案，或了解如何在職場與同事們共處。",
}

app = Flask(__name__)

line_bot_api = LineBotApi(os.environ.get("CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.environ.get("CHANNEL_SECRET"))


@app.route("/", methods=["GET", "POST"])
def callback():
    if request.method == "GET":
        return "Hello Heroku"
    if request.method == "POST":
        signature = request.headers["X-Line-Signature"]
        body = request.get_data(as_text=True)

        try:
            handler.handle(body, signature)
        except InvalidSignatureError:
            abort(400)

        return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    get_message = event.message.text
    haveAnswered = False
    for question in QA.keys():
        if question in get_message:
            reply = TextSendMessage(text=QA[question])
            haveAnswered = True
    if not haveAnswered:
        if "問問題" in get_message:
            actions = []
            for i,question in enumerate(QA.keys()):
                if i == 0:
                    continue
                actions.append(
                    MessageAction(label=f"{question}", text=f"{question}")
                )
            buttons_template = ButtonsTemplate(
                title="請問你要問什麼問題呢？", text="請選擇一個問題詢問～", actions=actions
            )
            reply = TemplateSendMessage(
                alt_text="Buttons alt text", template=buttons_template
            )
        else:
            reply = TextSendMessage(text=f"{get_message}")
    # Send To Line
    line_bot_api.reply_message(event.reply_token, reply)
