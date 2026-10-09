import torch
from torch.utils.data import Dataset, DataLoader


class TC_Data(Dataset):
    def __init__(self, data_path, tokenizer, new_id_convert, max_len = 512):

        self.tokenizer = tokenizer
        self.texts, self.labels = [], []
        self.max_len = max_len

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                ss_line = line.strip().split('_!_')
                self.texts.append(ss_line[3])
                self.labels.append(new_id_convert[ss_line[1]])

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
    
    def __len__(self):
        return len(self.texts)

    def collate_fn(self, batch):
        texts, labels = [], []
        for data in batch:
            texts.append(data[0])
            labels.append(data[1])

        tokens = self.tokenizer(
            texts,
            max_length = self.max_len,
            padding = 'longest',
            truncation = True,
            return_tensors = 'pt'
        )
        labels = torch.tensor(labels, dtype=torch.long)

        #input_ids, token_type_ids， attention_mask后添加labels
        tokens["labels"] = labels

        return tokens

    def get_dataLoader(self, batch_size, shuffle, num_workers, pin_memory):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn = self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import json
    from utils import args_analyse
    
    config = args_analyse()
    with open(config['data']['TC_new_id_convert_path'], "r", encoding="utf-8") as f:
        data = json.load(f)

    new_id_convert = data['new_id_convert']

    tokenizer_path = config['data']['tokenizer_path']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_datas = TC_Data(config['data']['data_path']['train'], tokenizer, new_id_convert)
    batch_size = 5
    shuffle = True
    num_workers = 4
    pin_memory = True
    dataloader = train_datas.get_dataLoader(batch_size, shuffle, num_workers, pin_memory)
    for batch in dataloader:
        print(batch)
        break