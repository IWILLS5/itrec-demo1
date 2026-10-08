import json
import os 

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
import swanlab

from config import Config
from model import TCModel_formBert
from datasets import TC_Data, TC_DataLoader
from datasets_clean import TC_clean
from evaluationMetrics import EvaluationMetrics


# 计算outputs中的文件夹数量
def count_folder(dir):
    folders = os.listdir(dir)
    return len(folders)

class TC_Experiment:
    def __init__(self, config, new_id_convert):
        self.config = config
        self.new_id_convert = new_id_convert
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TCModel_formBert(config.model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)
        self.datasets_len, self.dataloader = self.get_DataLoader(config.data, self.new_id_convert, self.tokenizer)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = config.train.epochs
        self.em = EvaluationMetrics(self.config.model.num_labels)

        #不冻结bert-base-model的参数，但以较小的学习率
        self.optimizer = optim.AdamW([
            {'params':self.model.pre_trained_model.parameters(), 'lr':config.train.adamW_bert_lr},
            {'params':self.model.classifier.parameters(), 'lr':config.train.adamW_classifier_lr}
        ], weight_decay=0.01, eps=1e-8)

        #统计当前outputs/demo1/中有多少
        nums = str(count_folder(config.exp.output_dir))
        os.mkdir(config.exp.output_dir / nums )

        self.best_check_point_path = config.exp.output_dir / nums / 'best_model.pt'
        self.history_path = config.exp.output_dir / nums / 'history.json'
        self.result_path = config.exp.output_dir / nums / 'result.json'
    
    #保存历史数据
    def save_history(self, history, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def _save_best_checkpoint(self, epoch, dev_acc):
        
        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'idx_to_class' : self.new_id_convert,
            'dev_acc' : dev_acc
        }
        torch.save(checkpoint, self.best_check_point_path)
    # 训练一个epoch
    def train_epoch(self, model, dataLoader, criterion, optimizer, device):
        
        model.train()
        training_loss = 0.0
        training_correct = 0
        training_samples = 0

        for batch in dataLoader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            output = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            training_loss += loss.item() * batch_size
            training_correct += (output.argmax(dim=1) == labels).sum().item()
            training_samples += batch_size

        #返回batch的平均loss和准确率
        return training_loss / training_samples, training_correct / training_samples
    #进行dev和test的评估
    @torch.inference_mode() #比torch.no_grad()更强
    def evaluate(self, model, dataLoader, criterion, evaluationMatrics, device):
        model.eval()

        evaluate_loss = 0.0
        evaluate_correct = 0
        evaluate_samples = 0

        for batch in dataLoader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(output, labels)

            # 指标类按样本累计整个验证集，而不是只处理一个样本
            predictions = output.argmax(dim=-1)
            evaluationMatrics.add_batch(
                labels.detach().cpu().tolist(),
                predictions.detach().cpu().tolist()
            )

            batch_size = labels.size(0)
            evaluate_loss += loss.item() * batch_size
            evaluate_correct += (output.argmax(1) == labels).sum().item()
            evaluate_samples += batch_size

        #返回batch的平均loss和准确率
        return evaluate_loss / evaluate_samples, evaluate_correct / evaluate_samples

    #获取三个数据集的数据加载器
    def get_DataLoader(self, config, new_id_convert, tokenizer):
        datasets = []
        datasets_len = {}
        for name, path in config.data_path.items():
            dataset = TC_Data(data_path=path, tokenizer=tokenizer,
                            new_id_convert=new_id_convert,
                            max_len=config.max_len)
            datasets_len[name] = len(dataset)
            datasets.append(dataset)
        train_datas, dev_datas, test_datas = datasets
        DataLoader_Factory = TC_DataLoader(tokenizer)

        train_DataLoader = DataLoader_Factory.get_dataLoader(
                train_datas, config.batch_size, True, config.num_workers,  config.pin_memory)

        dev_DataLoader = DataLoader_Factory.get_dataLoader(
                dev_datas, config.batch_size, False, config.num_workers,  config.pin_memory)

        test_DataLoader = DataLoader_Factory.get_dataLoader(
                test_datas, config.batch_size, False, config.num_workers,  config.pin_memory)
        return datasets_len ,{
            'train' : train_DataLoader, 
            'dev' : dev_DataLoader,
            'test' : test_DataLoader
        }
    
    #训练
    def train(self):

        # 初始化一个新的 swanlab 实验
        swanlab.init(
            # 设置将记录此次实验的项目信息
            project="itrec-tc",
            workspace="iwills",
            # 跟踪超参数和实验元数据
            config={
                "adamW_bert_lr": self.config.train.adamW_bert_lr,
                "adamW_classifier_lr": self.config.train.adamW_classifier_lr,
                "epochs": self.epochs,
                "dropout" : self.config.model.dropout,
                "batch_size" : self.config.data.batch_size
            }
        )

        
        history = {
            'train_loss' : [],
            'train_acc' : [],
            'dev_loss' : [],
            'dev_acc' : [],
            'dev_precision' : [],
            'dev_recall' : [],
            'dev_f1' : []
        }

        best_dev_acc = 0.0

        print('----------------配置展示----------------')
        print(f'运行设备:{self.device}')
        for name, size in self.datasets_len.items():
            print(f'{name}数据集的长度={size}')

        for epoch in range(1, self.epochs + 1):
            
            train_loss, train_acc = self.train_epoch(
                self.model,
                self.dataloader['train'],
                self.criterion,
                self.optimizer,
                self.device
            )

            #先将指标内容清零
            self.em.reset()
            dev_loss, dev_acc = self.evaluate(
                self.model,
                self.dataloader['dev'],
                self.criterion,
                self.em,
                self.device
            )
            
            dev_precision, dev_recall, dev_f1 = self.em.get_Macro_data()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['dev_loss'].append(dev_loss)
            history['dev_acc'].append(dev_acc)
            history['dev_precision'].append(dev_precision)
            history['dev_recall'].append(dev_recall)
            history['dev_f1'].append(dev_f1)

            swanlab.log({
                'train_loss' : train_loss,
                'train_acc' : train_acc,
                'dev_loss' : dev_loss,
                'dev_acc' : dev_acc,
                'dev_precision' : dev_precision, 
                'dev_recall' : dev_recall, 
                'dev_f1' : dev_f1, 
            })
            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Dev Loss: {dev_loss:.4f} | "
                f"Dev Acc: {dev_acc:.4f} | "
                f"Dev Precision: {dev_precision:.4f} | "
                f"Dev Recall: {dev_recall:.4f} | "
                f"Dev F1: {dev_f1:.4f}"
            )

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                self._save_best_checkpoint(epoch, dev_acc)
        self.save_history(history, self.history_path)

    def test_best_model(self):
        checkpoint = torch.load(
            self.best_check_point_path,
            map_location=self.device,
            weights_only=True
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        #先将指标内容清零
        self.em.reset()
        test_loss, test_acc = self.evaluate(
            self.model,
            self.dataloader['test'],
            self.criterion,
            self.em,
            self.device
        )

        precision, recall, f1 = self.em.get_Macro_data()

        result = {
            "best_epoch" : int(checkpoint['epoch']),
            "bese_dev_acc" : float(checkpoint['dev_acc']),
            "test_loss" : test_loss,
            "test_acc" : test_acc,
            "precision" : precision,
            "recall" : recall,
            "f1" : f1,
        }

        with open(self.result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent = 2)
        print(
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} "
            f"(best epoch: {result['best_epoch']})"
        )
        # resolve()打印绝对路径
        print(f"最佳模型已保存至：{self.best_check_point_path.resolve()}")

    def run(self):
        self.train()
        self.test_best_model()

def main():
    config = Config()
    #获取new_id_convert
    with open(config.data.TC_new_id_convert_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    new_id_convert = data['new_id_convert']
    exp = TC_Experiment(config, new_id_convert)
    exp.run()

if __name__ == '__main__':
    #包含数据、模型等超参数
    main()
