import requests
import json
import os
from linebot import LineBotApi

token = os.environ.get("CHANNEL_ACCESS_TOKEN")
line_bot_api = LineBotApi(f"{token}")

# clear richimage
rich_menu_list = line_bot_api.get_rich_menu_list()

for rich_menu in rich_menu_list:
    line_bot_api.delete_rich_menu(rich_menu.rich_menu_id)

# setting for rich menu
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
body = {
    "size": {"width": 2500, "height": 1686},
    "selected": "true",
    "name": "Controller",
    "chatBarText": "Controller",
    "areas": [
        {
            "bounds": {"x": 0, "y": 0, "width": 833, "height": 842},
            "action": {"type": "message", "text": "自我介紹"},
        },
        {
            "bounds": {"x": 833, "y": 0, "width": 833, "height": 842},
            "action": {"type": "message", "text": "問問題"},
        },
        {
            "bounds": {"x": 1666, "y": 0, "width": 833, "height": 842},
            "action": {"type": "uri", "uri": "https://pda.104.com.tw/profile/share/8pkKfdOd45rkB9uUinyZZt7SFmWb1img"},
        },
        {
            "bounds": {"x": 0, "y": 842, "width": 833, "height": 842},
            "action": {"type": "uri", "uri": "https://github.com/morris1961"},
        },
        {
            "bounds": {"x": 833, "y": 842, "width": 833, "height": 842},
            "action": {"type": "message", "text": "nope"},
        },
        {
            "bounds": {"x": 1666, "y": 842, "width": 833, "height": 842},
            "action": {"type": "message", "text": "nope"},
        },
    ],
}

# get menuId
req = requests.request(
    "POST",
    "https://api.line.me/v2/bot/richmenu",
    headers=headers,
    data=json.dumps(body).encode("utf-8"),
)
id = req.text[15:-2]
print(id)

# set image
with open("./img/test.png", "rb") as f:
    line_bot_api.set_rich_menu_image(id, "image/png", f)

# use rich menu
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
req = requests.request(
    "POST", f"https://api.line.me/v2/bot/user/all/richmenu/{id}", headers=headers
)
print(req.text)
