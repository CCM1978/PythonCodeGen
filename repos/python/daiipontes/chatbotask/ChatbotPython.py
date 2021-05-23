import sys

def processar_resposta(resposta):
 
  if resposta == '1':
        print ('No nosso site de sáude você pode acompanhar todas as noticias do por você e de todo o calendário de saúde, além disso pode se inscrever no nosso programa de acompanhamento nutricional\nLink do site')
  elif resposta == '2': 
        print ('\nEm casos de sintomas ligar para os telefones (11)xxxxx-xxxx ou (11)xxxxx-xxxx para ter o atendimento individual, eles funcionam 24 horas por dia e os atendimentos são feitos exclusivamente via ligação.\n')
  elif resposta == '3': 
        print ('\nOs ramais disponíveis no ambulatorio são:\n xxxx para reagendamentos e dúvidas\n xxxx para acidentes e emergências.\n')
  elif resposta == 'Sair': 
        print ('\nAgradeçemos o seu contato.\n')
        sys.exit()              
  elif resposta == 'Não': 
        print ('\nAgradecemos o seu contato.\n')
        sys.exit()    


def start():
 
  print ('Olá, Sou a assistente virtual do DSO, para começarmos com o seu atendimento responda as questões abaixo\n')

#matricula do funcionário substituia o nome 

  nome = input('Digite seu nome ')
  email =  input ('digite seu email ')

 
  while True:
 
      resposta = input (f'\nAgora é só escolher uma das opções abaixo\nLembrando que para encerrar o seu atendimento é só nos mandar a palavra: Sair\n1- Site de saúde;\n2- Informações de contato é Corona;\n3- Ramal Ambulatório;\n')
 
      processar_resposta(resposta)

      resposta = input ('\nPodemos ajudar em algo mais?\n')

      processar_resposta(resposta)
 
if __name__ == '__main__':
    start()
