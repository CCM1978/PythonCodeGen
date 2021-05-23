from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

button_options = KeyboardButton('/options')
button_want_to = KeyboardButton('/Хочу_рецепт_на_сегодня')
button_first = KeyboardButton('/Первые_блюда')
button_second = KeyboardButton('/Вторые_блюда')
button_dessert = KeyboardButton('/Десерты')
button_add_recipe = KeyboardButton('/Добавление_рецепта')

greet_kb = ReplyKeyboardMarkup(resize_keyboard=True).add(button_options)
greet_kb2 = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True
).add(button_want_to).add()
greet_kb3 = ReplyKeyboardMarkup(resize_keyboard=True).add(button_want_to).add(button_add_recipe).add(button_options)
greet_kb4 = ReplyKeyboardMarkup(resize_keyboard=True).add(button_first).add(button_second).add(button_dessert).add(button_options)
