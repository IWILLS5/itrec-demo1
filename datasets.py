import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
class TC_Data(Dataset):
    def __init__(self, data_path, tokenizer, new_id_convert, max_len = 512):

        self.tokenizer = tokenizer
        self.texts, self.labels = [], []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                ss_line = line.strip().split('_!_')
                self.texts.append(ss_line[3])
                self.labels.append(new_id_convert[ss_line[1]])

        #未进行max_len padding或者 longest，返回未list，等待collate_fn修改
        self.tokens = self.tokenizer(
            self.texts,
            max_length = max_len,
            padding = False,
            truncation = True,
            return_tensors = None
        )

        self.input_ids = self.tokens['input_ids']
        self.token_type_ids = self.tokens['token_type_ids']
        self.attention_mask = self.tokens['attention_mask']

    def __getitem__(self, index):
        return {
            'input_ids' : self.input_ids[index],
            'token_type_ids' : self.token_type_ids[index],
            'attention_mask' : self.attention_mask[index],
            'labels' : self.labels[index]
        }
    def __len__(self):
        return len(self.texts)

class TC_DataLoader():
    def __init__(self, tokenizer):

        self.pad_token_id = tokenizer.pad_token_id

    def collate_fn(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(data['input_ids'], dtype=torch.long) for data in batch],
            padding_value = self.pad_token_id, batch_first=True
        )
        token_type_ids = pad_sequence(
            [torch.tensor(data['token_type_ids'], dtype=torch.long) for data in batch],
            padding_value = self.pad_token_id, batch_first=True
        )
        attention_mask = pad_sequence(
            [torch.tensor(data['attention_mask'], dtype=torch.long) for data in batch],
            padding_value = self.pad_token_id, batch_first=True
        )
        labels = torch.tensor([data['labels'] for data in batch], dtype=torch.long)
        return {
            'input_ids' : input_ids,
            'token_type_ids' : token_type_ids,
            'attention_mask' : attention_mask,
            'labels' :labels
        }
    
    def get_dataLoader(self, datas, batch_size, shuffle, num_workers, pin_memory):
        return DataLoader(
            dataset=datas,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn = self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

if __name__ == '__main__':
    from transformers import AutoTokenizer
    import json
    from config import Config
    
    config = Config()
    with open(config.data.TC_new_id_convert_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_id_convert = data['new_id_convert']

    tokenizer_path = config.data.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    train_datas = TC_Data(config.data.data_path['train'], tokenizer, new_id_convert)
    batch_size = 5
    shuffle = True
    num_workers = 4
    pin_memory = True

    train_dataloader = TC_DataLoader(tokenizer).get_dataLoader(
        train_datas,
        batch_size=batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = pin_memory
    )

    first = True
    for batch in train_dataloader:
        # for data in batch:
        #     if first:
        #         print(data)
        #         break
        print(batch)
        break