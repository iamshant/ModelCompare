config = {
    'model1': 'roberta',
    'model2': 'bert',
    'tasks': {
        'multilabel': {
            'do_task': True,
            'ft': True,
            'epochs': 1,
            'dataset': 'joelito/sem_eval_2010_task_8',
            'text_column': 'sentence',
            'label_column': 'relation',
            'batch_size': 32,
            'learning_rate': 3e-5,
            'max_seq_len': 128,
            'distillation': False,
            'alpha': 0.1,
            'temperature': 2
        },
        'sentiment': {
            'do_task': False,
            'ft': True,
            'epochs': 1,
            'dataset': 'rotten_tomatoes',
            'text_column': 'text',
            'label_column': 'label',
            'batch_size': 32,
            'learning_rate': 3e-5,
            'max_seq_len': 128,
            'distillation': False,
            'alpha': 0.1,
            'temperature': 2
        },
        'qna': {
            'do_task': False,
            'ft': True,
            'epochs': 1,
            'dataset': 'squad',
            'batch_size': 32,
            'learning_rate': 3e-5,
            'max_seq_len': 128
        },
    }
}
