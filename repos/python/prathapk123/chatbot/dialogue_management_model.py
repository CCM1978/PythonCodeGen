from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import rasa_core
from rasa_core.agent import Agent
from rasa_core.channels import HttpInputChannel
from rasa_core.channels.console import ConsoleInputChannel,ConsoleOutputChannel
from rasa_core.interpreter import RegexInterpreter, NaturalLanguageInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter

logger = logging.getLogger(__name__)

def train_dialogue(domain_file = 'faq_domain.yml',
					model_path = './models/dialogue',
					training_data_file = 'data/stories.md'):
					
	agent = Agent(domain_file, policies = [KerasPolicy()])
	
	agent.train(
				training_data_file,
				epochs = 300,
				batch_size = 1,
				validation_split = 0.2)
				
	agent.persist(model_path)
	return agent

def run_faq_bot(serve_forever=True):
	interpreter = RasaNLUInterpreter('./models/nlu/default/faq_bot')
	agent = Agent.load('./models/dialogue', interpreter = interpreter)
	
	if serve_forever:
		agent.handle_channel(ConsoleInputChannel())
		#agent.handle_channel(HttpInputChannel(5004, "/app", ))

	return agent

if __name__ == '__main__':
	#train_dialogue()
	run_faq_bot()