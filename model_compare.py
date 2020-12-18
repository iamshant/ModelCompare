import os
import json
import time
import copy
import shutil
import warnings
from collections import defaultdict

from dataset import Dataset
from model import Model
from config import config

import tensorflow as tf
import tensorflow.keras.backend as K

import subprocess

class ModelCompare:
	'''
	Main class that controls the framework
	'''

	def __init__(self, model1, model2):
		'''
		Initializes model instances and results variable
		:param model1: name of first model
		:param model2: name of second model
		'''
		self.model1 = Model(model1)
		self.model2 = Model(model2)
		self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


	def __str__(self):
		'''
		Prints model names
		'''
		return 'Model 1: ' + self.model1.name + '\n' + 'Model 2: ' + self.model2.name + '\n'


	def run_tasks(self):
		'''
		Figures out which tasks to run from the configuration file and calls the appropriate methods
		'''
		if not os.path.exists('outputs'):
			os.makedirs('outputs')
		Model.prepare()
		TASK_MAP = {'sentiment': self.sentiment, 'multilabel': self.multilabel_classification, 'qna': self.qna}
		op_name = ''
		for task in config['tasks']:
			if config['tasks'][task]['do_task']:
				op_name = op_name + task + '_' + str(config['tasks'][task]['epochs']) + '_'
				TASK_MAP[task]()
		with open('outputs/' + op_name[:-1] + '.json', 'w') as fp:
		    json.dump(self.results, fp)


	def sentiment(self):
		'''
		Collects sentiment task params and runs classification method
		'''
		ft = config['tasks']['sentiment']['ft']
		dataset = config['tasks']['sentiment']['dataset']
		text_column = config['tasks']['sentiment']['text_column']
		label_column = config['tasks']['sentiment']['label_column']
		epochs = config['tasks']['sentiment']['epochs']
		learning_rate = config['tasks']['sentiment']['learning_rate']
		batch_size = config['tasks']['sentiment']['batch_size']
		max_seq_len = config['tasks']['sentiment']['max_seq_len']
		distil = config['tasks']['sentiment']['distillation']
		alpha = config['tasks']['sentiment']['alpha']
		temperature = config['tasks']['sentiment']['temperature']
		self.classification('sentiment', ft, dataset, epochs, batch_size, learning_rate, max_seq_len, distil, alpha, temperature, text_column, label_column)


	def multilabel_classification(self):
		'''
		Collects multilabel classification task params and runs classification method
		'''
		ft = config['tasks']['multilabel']['ft']
		dataset = config['tasks']['multilabel']['dataset']
		text_column = config['tasks']['multilabel']['text_column']
		label_column = config['tasks']['multilabel']['label_column']
		epochs = config['tasks']['multilabel']['epochs']
		learning_rate = config['tasks']['multilabel']['learning_rate']
		batch_size = config['tasks']['multilabel']['batch_size']
		max_seq_len = config['tasks']['multilabel']['max_seq_len']
		distil = config['tasks']['multilabel']['distillation']
		alpha = config['tasks']['multilabel']['alpha']
		temperature = config['tasks']['multilabel']['temperature']
		self.classification('multilabel', ft, dataset, epochs, batch_size, learning_rate, max_seq_len, distil, alpha, temperature, text_column, label_column)


	def classification(self, cls_type, ft, dataset, epochs, batch_size, learning_rate, max_seq_len, distil, alpha, temperature, text_column, label_column):
		'''
		Loads datasets, models and runs training and evaluation pipelines
		:param cls_type: classifictaion task type
		:param ft: whether finetuning should be done
		:param dataset: dataset name
		:param epochs: number of training epochs
		:param batch_size: batch size
		:param learning_rate: learning rate
		:param max_seq_len: maximum sequence length
		:param distil: whether to distil model
		:param alpha: weight for student loss
		:param temperature: temperature for softmax operation for distillation
		:param text_column: column name for text in dataset
		:param label_column: column name for label in dataset
		'''
		if ft:
			train_dataset = Dataset(dataset, split=Dataset.TRAIN_STR)
		val_dataset = Dataset(dataset, split=Dataset.VALIDATION_STR)

		for i in (self.model1, self.model2):
			opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
			loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
			metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]

			model = i.load_model(cls_type, train_dataset.get_num_classes(label_column=label_column), max_seq_len)
			model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

			if ft:
				tf_train_data = train_dataset.classification_tokenize(i.tokenizer, batch_size, max_seq_len,
																		i.name, text_column=text_column, label_column=label_column)
				model.fit(tf_train_data, epochs=epochs)

				if distil:
					model_teacher = i.load_model(cls_type, train_dataset.get_num_classes(label_column=label_column), max_seq_len, for_distillation=True, temperature=temperature)
					model_teacher.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
					model_teacher.fit(tf_train_data, epochs=epochs)
					model_soft_labels = model_teacher.predict(tf_train_data)
					student_dataset, encoder = train_dataset.student_dataset_encoder(model_soft_labels, batch_size, text_column=text_column, label_column=label_column)
					model_student = Model.student_model(cls_type, encoder, train_dataset.get_num_classes(label_column=label_column), temperature=temperature)

					student_loss_fn = {'soft': Model.get_distillation_loss_fn(), 'hard': Model.get_distillation_loss_fn()}
					loss_wts = {'soft': 1 - alpha, 'hard': alpha}
					model_student.compile(optimizer=opt, loss=student_loss_fn, loss_weights=loss_wts, metrics=metrics)

					model_student.fit(student_dataset, epochs=epochs)

					del model_soft_labels
					del student_dataset

				del tf_train_data

			tf_val_data = val_dataset.classification_tokenize(i.tokenizer, batch_size, max_seq_len,
																i.name, text_column=text_column, label_column=label_column)
			start = time.time()
			model_eval = {k:v for k, v in zip(model.metrics_names, model.evaluate(tf_val_data, verbose=0))}
			time_taken = time.time() - start
			self.results[cls_type][i.name] = model_eval
			try:
				self.results[cls_type][i.name]['f1'] = (2 * model_eval['precision'] * model_eval['recall']) / (model_eval['precision'] + model_eval['recall'])
			except ZeroDivisionError:
				self.results[cls_type][i.name]['f1'] = 0
			self.results[cls_type][i.name]['time taken'] = time_taken
			del self.results[cls_type][i.name]['precision']
			del self.results[cls_type][i.name]['recall']

			if distil:
				model_soft_labels = model.predict(tf_val_data)
				val_student_dataset, encoder = val_dataset.student_dataset_encoder(model_soft_labels, batch_size, text_column=text_column, label_column=label_column)
				start = time.time()
				model_eval = {k.split('_')[-1]:v for k, v in zip(model_student.metrics_names, model_student.evaluate(val_student_dataset, verbose=0)) if 'soft' not in k}
				time_taken = time.time() - start
				self.results[cls_type]['distilled-' + i.name] = model_eval
				try:
					self.results[cls_type]['distilled-' + i.name]['f1'] = (2 * model_eval['precision'] * model_eval['recall']) / (model_eval['precision'] + model_eval['recall'])
				except ZeroDivisionError:
					self.results[cls_type]['distilled-' + i.name]['f1'] = 0
				self.results[cls_type]['distilled-' + i.name]['time taken'] = time_taken
				del self.results[cls_type]['distilled-' + i.name]['precision']
				del self.results[cls_type]['distilled-' + i.name]['recall']

				del tf_val_data
				del val_student_dataset

			Model.clean_up()


	def qna(self):
		'''
		Collects question answering task params and runs qna task
		'''
		ft = config['tasks']['qna']['ft']
		dataset = config['tasks']['qna']['dataset']
		epochs = config['tasks']['qna']['epochs']
		batch_size = config['tasks']['qna']['batch_size']
		learning_rate = config['tasks']['qna']['learning_rate']
		max_seq_length = config['tasks']['qna']['max_seq_len']
		if dataset != 'squad':
			warning.warn('Only SQuAD is currently supported for QnA. Defaulting to SQuAD')
			dataset = 'squad'

		def get_command(model, do_train=True):
			command = ['python', 'qa_utils/run_qa.py', \
						'--model_name_or_path', model, \
						'--dataset_name', dataset, \
						'--do_eval', \
						'--per_device_train_batch_size', str(batch_size), \
						'--learning_rate', str(learning_rate), \
						'--num_train_epochs', str(epochs), \
						'--max_seq_length', str(max_seq_length), \
						'--doc_stride', '32', \
						'--output_dir', '/home/jupyter/ModelCompare/qna_output']
			if do_train:
				command.append('--do_train')
			return command

		if ft:
			command1 = get_command(Model.MODEL_MAP[self.model1.name][0])
			command2 = get_command(Model.MODEL_MAP[self.model2.name][0])
		else:
			command1 = get_command(Model.MODEL_MAP[self.model1.name][0], False)
			command2 = get_command(Model.MODEL_MAP[self.model2.name][0], False)

		p1 = subprocess.run(command1)
		try:
			shutil.rmtree('qna_output/')
			shutil.rmtree('runs/')
		except OSError as e:
			print("Folder deletion error")
		p2 = subprocess.run(command2)
		model1_name = Model.MODEL_MAP[self.model1.name][0].split('-')[0]
		model2_name = Model.MODEL_MAP[self.model2.name][0].split('-')[0]        
		with open('qna_results_' + model1_name + '.json') as f:
			d1 = json.load(f)
		with open('qna_results_' + model2_name + '.json') as f:
			d2 = json.load(f)
		self.results['qna'] = {model1_name: d1['qna'][model1_name], model2_name: d2['qna'][model2_name]}


if __name__ == '__main__':
	f = ModelCompare(config['model1'], config['model2'])
	f.run_tasks()