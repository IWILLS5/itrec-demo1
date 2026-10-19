import torch
import argparse
import json

default_configPath = "Demo/demo1/configs/base.json"

#获取训练超参数
def args_analyse():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default= default_configPath
    )
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"配置文件不存在: {args.config}")
        option = input("是否选择继续使用base.json配置(y or n):\n")
        if option == 'y':
            with open(default_configPath, 'r', encoding='utf-8') as f:
                        config = json.load(f)
        else:
            raise
    return config            


def fmt(x):
    if isinstance(x, int):
        return str(x)
    s = f"{x:e}"                       # 1e-3 → '1.000000e-03'
    mantissa, exp = s.split('e')
    mantissa = mantissa.rstrip('0.')  # '1.000000' → '1'
    exp = str(int(exp))                # '-03' → '-3'
    return f"{mantissa}e{exp}"

def op_parse(output_folder, config):
    parts = [
        str(config['train']['epochs']), 
        str(config['data']['batch_size']),
        str(config['model']['dropout']),
        fmt(config['train']['adamW_bert_lr']),
        fmt(config['train']['adamW_classifier_lr'])
    ]

    return output_folder/ '_'.join(parts)


def seed_init(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)