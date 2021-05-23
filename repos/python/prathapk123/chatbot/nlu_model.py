#from rasa_nlu.training_data import load_data
import rasa_nlu
from rasa_nlu.converters import load_data
from rasa_nlu import config
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter

def train_nlu(data, configs_nm, model_dir):
	training_data = load_data(data)
	trainer = Trainer(RasaNLUConfig(configs_nm))
	trainer.train(training_data)
	model_directory = trainer.persist(model_dir, fixed_model_name = 'faq_bot')
	
def run_nlu():
	interpreter = Interpreter.load('./models/nlu/default/faq_bot')
	print(interpreter.parse("Hello!"))
	
if __name__ == '__main__':
	train_nlu('./data/data.json', 'config_spacy.json', './models/nlu')
	run_nlu()