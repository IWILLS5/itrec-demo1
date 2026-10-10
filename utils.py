import argparse
import json

#获取训练超参数
def args_analyse():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="Demo/demo1/configs/base.json"
    )
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
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