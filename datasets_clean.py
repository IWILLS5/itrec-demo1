import json


def TC_clean(data_path, TC_new_id_convert_path):
    datas = {}
    new_id = set()
    new_id_label = {}
    new_id_convert, id_to_new_id = {}, {}
    for name, path in data_path.items():
        temp_datas = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                ss_line = line.strip().split('_!_')
                if (ss_line[1], ss_line[2]) not in new_id:
                    new_id.add((ss_line[1], ss_line[2]))
                temp_datas.append(ss_line)
            datas[name] = temp_datas
    print('--------------------------------------')
    for name in datas.keys():
        print(f'{name}数据集的长度:{len(datas[name])}')

    for k, v in enumerate(sorted(new_id, key=lambda x : x[0])):
        new_id_convert[v[0]] = k
        id_to_new_id[k] = v[0]
        new_id_label[v[0]] = v[1]

    with open(TC_new_id_convert_path, 'w', encoding='utf-8') as f:
        json.dump({
            "new_id_label" : new_id_label,
            "new_id_convert" : new_id_convert
        }, f, ensure_ascii=False, indent=2)

    def re_ids_find_max_len(datas):
        new_ids = []
        max_text_len = -1
        sorts = [0] * len(id_to_new_id.keys())
        for data in datas:
            sorts[new_id_convert[data[1]]] += 1
            new_ids.append(data[0])
            text_len = len(data[3])
            if text_len > max_text_len:
                max_text_len = text_len
        return new_ids, max_text_len, sorts

    train_new_ids, max_train_text_len, train_sorts = re_ids_find_max_len(datas['train'])
    dev_new_ids, max_dev_text_len, dev_sorts = re_ids_find_max_len(datas['dev'])
    test_new_ids, max_test_text_len, test_sorts = re_ids_find_max_len(datas['test'])
    
    print('--------------------------------------')
    print(f'train数据集文本最大长度 = {max_train_text_len}')
    print(f'dev数据集文本最大长度 = {max_dev_text_len}')
    print(f'test数据集文本最大长度 = {max_test_text_len}')

    print('----------------train各个类别数据数量----------------')
    for i, nums in enumerate(train_sorts):
        print(f'第{id_to_new_id[i]}类有 {nums}')
    print('----------------dev各个类别数据数量----------------')
    for i, nums in enumerate(dev_sorts):
        print(f'第{id_to_new_id[i]}类有 {nums}')
    print('----------------test各个类别数据数量----------------')
    for i, nums in enumerate(test_sorts):
        print(f'第{id_to_new_id[i]}类有 {nums}')

    train_new_ids_set = set(train_new_ids)
    dev_new_ids_set = set(dev_new_ids)
    test_new_ids_set = set(test_new_ids)

    if len(train_new_ids_set) != len(datas['train']):
        print('train数据集有重复数据!!!')
    if len(dev_new_ids_set) != len(datas['dev']):
        print('dev数据集有重复数据!!!')
    if len(test_new_ids_set) != len(datas['test']):
        print('test数据集有重复数据!!!')

    print('------------train数据清洗-------------')
    repeat_ids_dev, repeat_ids_test = [], []
    for id in train_new_ids:
        if id in dev_new_ids_set:
            repeat_ids_dev.append(id)
        if id in test_new_ids_set:
            repeat_ids_test.append(id)
    print(f'repeat_ids_dev : {repeat_ids_dev}')
    print(f'repeat_ids_test : {repeat_ids_test}')

    print('------------dev数据清洗-------------')
    repeat_ids_test = []
    for id in dev_new_ids:
        if id in repeat_ids_test:
            repeat_ids_dev.append(id)
    print(f'repeat_ids_test : {repeat_ids_test}')


if __name__ == '__main__':
    from config import Config
    config = Config()
    data_path = config.data.data_path
    TC_new_id_convert_path = config.data.TC_new_id_convert_path
    
    TC_clean(data_path, TC_new_id_convert_path)