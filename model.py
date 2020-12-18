from transformers import logging
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModelForQuestionAnswering

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Activation, Embedding, Bidirectional, LSTM, Dense, Lambda, BatchNormalization, Dropout

class Model:
	'''
	Loads main model and student model, prepares framework for training and clears memory when done
	'''


	MODEL_MAP = {'bert': ('bert-base-uncased', ''), 'roberta': ('roberta-base', ''), 'xlnet': ('xlnet-base-cased', '')}

	MODEL_INPUTS = {'bert': ['input_ids', 'token_type_ids', 'attention_mask'], 
					'roberta': ['input_ids', 'attention_mask'], 'xlnet': ['input_ids', 'attention_mask']}


	def __init__(self, name):
		'''
		Initializes model classes
		:param name: model name
		'''
		self.name = name
		self.classification_model = TFAutoModelForSequenceClassification
		self.qna_model = TFAutoModelForQuestionAnswering
		self.type = self.MODEL_MAP[self.name]
		self.tokenizer = AutoTokenizer.from_pretrained(self.type[0])


	@staticmethod
	def student_model(task, encoder, num_classes, temperature=2):
		'''
		Creates student model depending on task
		:param encoder: student model encoder
		:param num_classes: number of classes in datatset
		:temperature: softmax temperature for distillation
		:return: student model
		'''
		if task == 'sentiment':
			x = Input(shape=(1,), dtype=tf.string)
			y = encoder(x)
			y = Embedding(
			        input_dim=len(encoder.get_vocabulary()),
			        output_dim=64,
			        mask_zero=True)(y)
			y = Bidirectional(LSTM(64))(y)
			y = Dense(64, activation='relu')(y)
			y = BatchNormalization()(y)
			y = Dropout(0.2)(y)
			y = Dense(num_classes)(y)
			y1 = Lambda(lambda x: x / temperature)(y)
			y1 = Activation('softmax', name='soft')(y1)
			y2 = Activation('softmax', name='hard')(y)

			model = tf.keras.Model(x, [y1, y2])
			return model
		if task == 'multilabel':
			x = Input(shape=(1,), dtype=tf.string)
			y = encoder(x)
			y = Embedding(
			        input_dim=len(encoder.get_vocabulary()),
			        output_dim=64,
			        mask_zero=True)(y)
			y = Bidirectional(LSTM(64))(y)
			y = Dense(64, activation='relu')(y)
			y = BatchNormalization()(y)
			y = Dropout(0.2)(y)
			y = Dense(num_classes)(y)
			y1 = Lambda(lambda x: x / temperature)(y)
			y1 = Activation('sigmoid', name='soft')(y1)
			y2 = Activation('sigmoid', name='hard')(y)

			model = tf.keras.Model(x, [y1, y2])
			return model


	@staticmethod
	def get_distillation_loss_fn():
		'''
		Creates cross entropy loss objective for distillation process
		'''
		def loss_fn(y_true, y_pred):
			y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
			loss = -1 * (1 - y_pred) * y_true * K.log(y_pred) - y_pred * (1 - y_true) * K.log(1 - y_pred)
			return loss
		return loss_fn

		
	@staticmethod
	def prepare():
		'''
		Sets GPU memory growth
		'''
		try:
			gpu_list = tf.config.list_physical_devices('GPU')
			for gpu in gpu_list:
				tf.config.experimental.set_memory_growth(gpu, True)
			tf.config.set_soft_device_placement(True)
		except RuntimeError:
			pass


	@staticmethod
	def clean_up():
		'''
		Cleans up session after task execution
		'''
		K.clear_session()


	def load_model(self, task, num_classes, max_seq_len, for_distillation=False, temperature=1):
		'''
		Loads model based on task
		:param task: name of task
		:param num_classes: number of classes from dataset
		:param max_seq_len: maximum sequence length
		:param for_distillation: whether model is being loaded as teacher model
		:param temperature: softamx temperature for distillation
		:return: language model loaded from Huggingface's transformers library
		'''
		if task == 'sentiment':
			inputs = []
			for ip in self.MODEL_INPUTS[self.name]:
				inputs.append(Input((max_seq_len,), dtype='int32', name=ip))
			base_model = self.classification_model.from_pretrained(self.type[0], num_labels=num_classes,
															output_attentions=False,
															output_hidden_states=False)
			y = base_model(inputs)[0]
			if for_distillation:
				y = Lambda(lambda x: x / temperature)(y)
			y = Activation('softmax')(y)

			model = tf.keras.Model(inputs, y)
			return model
		elif task == 'multilabel':
			inputs = []
			for ip in self.MODEL_INPUTS[self.name]:
				inputs.append(Input((max_seq_len,), dtype='int32', name=ip))
			base_model = self.classification_model.from_pretrained(self.type[0], num_labels=num_classes,
															output_attentions=False,
															output_hidden_states=False)
			y = base_model(inputs)[0]
			if for_distillation:
				y = Lambda(lambda x: x / temperature)(y)
			y = Activation('sigmoid')(y)

			model = tf.keras.Model(inputs, y)
			return model
