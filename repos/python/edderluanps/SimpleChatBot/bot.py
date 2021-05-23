from nltk.chat.util import Chat, reflections

pairs = [

    [
        r'Olá!',
        ['olá!', 'opa!', 'oi, tudo bem?']
    ],
[
        r'Vamos sim!',
        ['olá!', 'opa!', 'oi, tudo bem?']
    ],
    [
        r'Qual o seu nome?',
        ['ChatBot', '...']
    ],
    [
        r'qual sua idade?',
        ['Alguns Dias', 'curioso...']
    ],
    [
        r'tchau?',
        ['tchau, até a próxima :D', 'adeus']
    ],
]

def bot_bot():
    print('ola! vamos conversar?')
    chat = Chat(pairs, reflections)
    chat.converse()

if __name__== '__main__':
    bot_bot()