import torch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_path = {
        'train': 'datas/0.demo1文本分类/train_3k.txt',
        'dev': 'datas/0.demo1文本分类/dev_1k.txt',
        'test': 'datas/0.demo1文本分类/test_1k.txt'
    }
    tokenizer_path = './pretrained_models/bert-base-chinese'
    max_len = 512
    batch_size = 32
    num_workers = 4
    pin_memory = True if torch.cuda.is_available() else False
@dataclass
class ModelConfig:
    pt_model_path = "./pretrained_models/bert-base-chinese"
    num_labels = 15
    dropout = 0.1

@dataclass
class TrainingConfig():
    epochs = 3
    adamW_bert_lr = 2e-5
    adamW_classifier_lr = 1e-3

@dataclass
class ExperimentConfig():
    output_dir = Path("outputs/demo1")

@dataclass
class Config:
    data:DataConfig = field(default_factory=DataConfig)
    model:ModelConfig = field(default_factory=ModelConfig)
    train:TrainingConfig = field(default_factory=TrainingConfig)
    exp:ExperimentConfig = field(default_factory=ExperimentConfig)