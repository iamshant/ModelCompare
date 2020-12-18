import os
from datasets import load_dataset, list_datasets, logging

import tensorflow as tf

from model import Model

import numpy as np

import copy

class Dataset:
	'''
	Loads datasets and tokenizes them
	'''

	HF_DATASETS = list_datasets()
	DATA_PATH = '../data/'

	TRAIN_STR = 'train'
	TEST_STR = 'test'
	VALIDATION_STR = 'validation'

	def __init__(self, name, split):
		'''
		Initialzes dataset
		:param name: name of dataset
		:param split: train/validation/test split
		'''
		self.name = name
		self.split = split
		if self.name not in self.HF_DATASETS:
			self.type = 'csv'
		else:
			self.type = 'hf'

		self.data = self.get_dataset()


	def get_num_classes(self, label_column='label'):
		'''
		Fetches number of classes in dataset
		:return: number of classes in dataset
		'''
		return self.data.features[label_column].num_classes


	def get_dataset(self):
		'''
		Loads dataset from Huggingface repository
		'''
		if self.type == 'hf':
			if self.split == self.VALIDATION_STR:
				try:
					return load_dataset(self.name, split=self.VALIDATION_STR)
				except ValueError:
					pass
				try:
					return load_dataset(self.name, split=self.TEST_STR)
				except ValueError:
					raise RuntimeError('Invalid dataset. No validation set found.')
			else:
				return load_dataset(self.name, split=self.split)
		else:
			filename = os.path.join(self.DATA_PATH, self.name, str(self.split) + '.' + str(self.type))
			return load_dataset(self.type, data_files=filename)


	def student_dataset_encoder(self, soft_labels, batch_size, text_column='text', label_column='label'):
		'''
		Creates student dataset in tf.Dataset format along with student model encoder
		:param soft_labels: soft labels from teacher model
		:param batch_size: batch_size
		:param text_column: column name for text in dataset
		:param label_column: column name for label in dataset
		:return: student dataset and student model encoder
		'''
		dataset = copy.deepcopy(self.data)
		dataset.set_format(type='tensorflow', columns=[text_column])
		features = dataset[text_column]
		hard_labels = tf.keras.utils.to_categorical(dataset[label_column], num_classes=self.get_num_classes(label_column=label_column))
		labels = {'soft': soft_labels, 'hard': hard_labels}
		tfdataset = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(self.data.num_rows).batch(batch_size)
		VOCAB_SIZE = 30522
		encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
		    max_tokens=VOCAB_SIZE)
		encoder.adapt(tfdataset.map(lambda text, label: text))

		return tfdataset, encoder


	def classification_tokenize(self, tokenizer, batch_size, max_seq_len, model_name, text_column='text', label_column='label'):
		'''
		Tokenizes data for classification task
		:param tokenizer: tokenizer class
		:param batch_size: batch_size
		:param max_seq_len: maximum sequence length
		:param model_name: model name
		:return: tokenized data
		'''
		def encode(example):
			return tokenizer(example[text_column], padding='max_length', truncation=True)
		dataset = self.data.map(encode)
		dataset.set_format(type='tensorflow', columns=Model.MODEL_INPUTS[model_name]+[label_column])
		features = {x: dataset[x].to_tensor(default_value=0, shape=(None, max_seq_len)) for x in Model.MODEL_INPUTS[model_name]}
		labels = tf.keras.utils.to_categorical(dataset[label_column], num_classes=self.get_num_classes(label_column=label_column))
		tfdataset = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(self.data.num_rows).batch(batch_size)
		return tfdataset
