# ModelCompare
A framework to compare 2 large language models.

Models supported:
1. BERT
2. XLNet
3. RoBERTa.

Tasks supported:
1. Multilabel Classification
2. Sentiment Classification
3. Question Answering

Default datasets:
1. Multilabel Classification
2. Sentiment Classification
3. Question Answering

Training hyperparameters like batch size and learning rate can be changed in [config.py](https://github.com/adityadesai97/ModelCompare/blob/main/config.py)

Classification tasks also support a distillation feature. Distillation hyperparameters can be changed in [config.py](https://github.com/adityadesai97/ModelCompare/blob/main/config.py)

# Installation:
```
git clone https://github.com/adityadesai97/ModelCompare.git
```

# Usage:
## Configuring:
An example configuration is in [config.py](https://github.com/adityadesai97/ModelCompare/blob/main/config.py)

All values are required. Leave default values if you do not want to vary them.

Hyperparameters:

1. Common:
    1. do_task
    2. ft
    3. epochs
    4. dataset
    5. batch_size
    6. learning_rate
    7. max_seq_len
2. Multilabel Classification / Sentiment Analysis:
    1. text_column
    2. label_column
    3. distillation
    4. alpha
    5. temperature
## Running the code:
```
python model_compare.py
```
