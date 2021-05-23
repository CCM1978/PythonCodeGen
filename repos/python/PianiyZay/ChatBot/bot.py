from random import randint
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from config import TOKEN
import keyboards as kb

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


#старт бота
@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Привет! Я бот и помогу выбрать блюдо на сегодня, и не только!", reply_markup=kb.greet_kb)


# @dp.message_handler()
# async def question(msg: types.Message):
#     await bot.send_message(msg.from_user.id, 'Выберите блюдо на сегодня', reply_markup=kb.greet_kb4)


#Список всех комманд
@dp.message_handler(commands=['options'])
async def process_hi2_command(message: types.Message):
    await message.reply("Что хотите?\n1.Выбор блюда\n2.Добавление блюда(don't use)", reply_markup=kb.greet_kb3)


#Выбор блюда
@dp.message_handler(commands=['Хочу_рецепт_на_сегодня'])
async def what_today(message: types.Message):
    await message.reply("Что хотите? Первое, второе, а может быть десерт?", reply_markup=kb.greet_kb4)
    
    
#Первое второе или десерт?
@dp.message_handler(commands=['Первые_блюда'])
async def first_recipe(message: types.Message):
    await message.reply(rand_recipe(), reply_markup=kb.greet_kb4)


def rand_recipe():
    a = randint(0, len(db_first) - 1)
    rand_some = str(db_first[a]['name'] + '\n' + db_first[a]['recipe'])
    return rand_some


db_first = [
    dict(name='Борщ', recipe='Какой-то рецепт'),
    dict(name='Солянка', recipe='Какой-то рецепт'),
    dict(name='Гороховый суп', recipe='Какой-то рецепт')
]


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
