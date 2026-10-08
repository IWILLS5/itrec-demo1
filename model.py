import torch.nn as nn
from transformers import BertModel

class TCModel_formBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        pt_model_path = config.pt_model_path
        num_labels = config.num_labels
        dropout = config.dropout
        
        #预训练模型
        self.pre_trained_model = BertModel.from_pretrained(
            pretrained_model_name_or_path = pt_model_path, add_pooling_layer = False)
        self.hidden_size = self.pre_trained_model.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        #分类头
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        
    def forward(self, input_ids, token_type_ids, attention_mask):
        #bert主干输出的内容
        output = self.pre_trained_model(
            input_ids = input_ids,
            token_type_ids = token_type_ids, 
            attention_mask = attention_mask
        )
        #取出CLS
        cls_output = output.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits


if __name__ == '__main__':
    pt_model_path = "./pretrained_models/bert-base-chinese"
    num_labels = 15
    model = TCModel_formBert(pt_model_path, num_labels)
    print(model)