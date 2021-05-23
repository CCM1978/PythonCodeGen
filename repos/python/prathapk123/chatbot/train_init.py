from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

#from rasa_core.featurizers import (MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer)
from rasa_core.featurizers import BinaryFeaturizer
if __name__ == '__main__':
	logging.basicConfig(level='INFO')
	
	training_data_file = './data/stories.md'
	model_path = './models/dialogue'
	
	featurizer = BinaryFeaturizer()
	#agent = Agent('faq_domain.yml', policies = [MemoizationPolicy(max_history=2), KerasPolicy(featurizer)])
	agent = Agent('faq_domain.yml', policies=[KerasPolicy(featurizer)])
	#training_data = agent.load_data(training_data_file)
	agent.train(
			training_data_file,
			augmentation_factor = 50,
			epochs = 100,
			batch_size = 1,
			validation_split = 0.2)
			
	agent.persist(model_path)
