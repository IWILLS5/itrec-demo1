from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

class TC_Data(Dataset):
    def __init__(self, data_path, tokenizer, new_id_convert, max_len = 512):

        self.tokenizer = tokenizer
        self.texts, self.labels = [], []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                ss_line = line.strip().split('_!_')
                self.texts.append(ss_line[3])
                self.labels.append(new_id_convert[ss_line[1]])
        
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
        self.collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding='longest',
            pad_to_multiple_of=8
        )
    def get_dataLoader(self, datas, batch_size, shuffle, num_workers, pin_memory):
        return DataLoader(
            dataset=datas,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn = self.collator,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

if __name__ == '__main__':
    from transformers import AutoTokenizer
    new_id_convert = {
        '100' : 0, '101' : 1, '102' : 2, '103' : 3, 
        '104' : 4, '106' : 5, '107' : 6, '108' : 7, 
        '109' : 8, '110' : 9, '112' : 10, '113' : 11, 
        '114' : 12, '115' : 13, '116' : 14 
    }
    
    data_path = {
        'train': 'datas/0.demo1文本分类/train_3k.txt',
        'dev': 'datas/0.demo1文本分类/dev_1k.txt',
        'test': 'datas/0.demo1文本分类/test_1k.txt'
    }
    tokenizer_path = 'pretrained_models/bert-base-chinese'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    train_datas = TC_Data(data_path['train'], tokenizer, new_id_convert)
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