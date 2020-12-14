// local variables
local max_turn_len=5;
local max_length=256;
local num_epochs=10;
local batch_size=64;
local bert_lr=3e-5;
local l1_reg=0.0; // l1正则化系数
local l2_reg=0.0; // l2正则化系数
local bert_path="/home/zs261988/models/ptms/bert_rbt3_pytorch/";
local train_data_path="/home/zs261988/data/rewrite/mask/mask_train.txt";      // 200k数据集  all_data/restore_train.txt   业务数据集：mask/mask_train.txt
local validation_data_path="/home/zs261988/data/rewrite/mask/mask_val.txt";
local device=0;
local model_size=768;  // tiny: 312     base: 768
local train_nums=35000;  // 229266
local weight_decay=0.0;
local warmup_steps=547;  // = 35000 / batch_size  64:547  32:1094  all: 3583
local seed=2020;

{
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "dataset_reader": {
        "type": "bert_span_pretrain",
        "model_name": bert_path,
        "max_turn_len": max_turn_len,
        "max_length": max_length,
        "lazy": false,
    },
    "model": {
        "type": "bert_span_pretrain",
        "model_name": bert_path,
        "max_turn_len": max_turn_len,
        "mask_task": true,
        "cls_task": true,
        "seed": seed,
        "initializer": {
            "regexes": [
                [".*weight",
                {
                    "type": "normal",
                    "mean": 0.01,
                    "std": 0.1
                 }],
                 [".*bias",
                 {
                    "type": "zero"
                 }]
            ]
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size
        }
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "weight_decay": weight_decay,
            "lr": bert_lr,
            "eps": 1e-8
        },
        "moving_average": {
            "type": "exponential"
        },
        "cuda_device": device,
        "patience": 5,
        "grad_norm": 10.0,
        "num_epochs": num_epochs,
        "validation_metric": "+cls_acc"
    }
}