import os
import argparse
import json

#获取训练超参数
def args_analyse():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="Demo/demo1/configs/base.json"
    )
    #data的数据)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    #model的数据
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--dropout", type=float)
    #train的数据
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--adamW_bert_lr", type=float)
    parser.add_argument("--adamW_classifier_lr", type=float)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)


    if args.max_len is not None:
        config["data"]["max_len"] = args.max_len

    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size

    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers

    if args.num_labels is not None:
        config["model"]["num_labels"] = args.num_labels

    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout

    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    if args.adamW_bert_lr is not None:
        config["train"]["adamW_bert_lr"] = args.adamW_bert_lr

    if args.adamW_classifier_lr is not None:
        config["train"]["adamW_classifier_lr"] = args.adamW_classifier_lr

    
    return config

# 计算dir中的文件夹数量
def count_folder(dir):
    folders = os.listdir(dir)
    return len(folders)
