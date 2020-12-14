// local variables
local max_turn_len=3;
local max_length=256;
local num_epochs=20;
local batch_size=64;
local bert_lr=3e-5;
local attention_lr=1e-4;
local neg_sample_ratio=0.0;
local l1_reg=0.0; // l1正则化系数
local l2_reg=0.0; // l2正则化系数
local bert_path="/home/zs261988/models/ptms/albert_void_tiny/";
local train_data_path="/home/zs261988/data/rewrite/mask/mask_train_word_expand.txt";
local validation_data_path="/home/zs261988/data/rewrite/mask/mask_val_word.txt";
local task_pretrained_file="/home/zs261988/models/mask_resolution/bert_rbt3_span_pretrain_cls_and_mask_bs_word/best.th";
local device=0;
local model_size=768;  // tiny: 312     base: 768
local train_nums=35000;
local weight_decay=0.0;
local warmup_steps=547;  // = 35000 / batch_size  64:547  32:1094
local seed=2020;

{
    "dataset_reader": {
        "type": "bert_word_span_resolution",
        "model_name": bert_path,
        "max_turn_len": max_turn_len,
        "max_length": max_length,
        "lazy": false,
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "model": {
        "type": "bert_span_pointer_resolution",
        "model_name": bert_path,
        "neg_sample_ratio": neg_sample_ratio,
        "seed": seed,
        "initializer": {
            "regexes": [
                [".*_attention.*weight",
                {
                    "type": "normal",
                    "mean": 0.01,
                    "std": 0.1
                 }],
                 [".*_attention.*bias",
                 {
                    "type": "zero"
                 }]
            ]
        },
        "regularizer": {
            "regexes": [
                [".*transformer_model.*weight",
                {
                    "type": "l2",
                    "alpha": l2_reg
                }],
                [".*transformer_model.*weight",
                {
                    "type": "l1",
                    "alpha": l1_reg
                }],
                [".*_attention.*weight",
                {
                    "type": "l2",
                    "alpha": l2_reg
                }],
                [".*_attention.*weight",
                {
                    "type": "l1",
                    "alpha": l1_reg
                }],
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
        "validation_metric": "+semr"
    }
}