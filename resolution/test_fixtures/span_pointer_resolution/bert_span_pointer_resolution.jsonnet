// local variables
local max_turn_len=3;
local max_length=30;
local num_epochs=1;
local batch_size=2;
local bert_lr=1e-4;
local attention_lr=1e-4;
local neg_sample_ratio=0.0;
local l1_reg=0.0; // l1正则化系数
local l2_reg=0.0; // l2正则化系数
local bert_path="adabrain/tests/data/base_bert/config.json";
local train_data_path="adabrain/resolution/test_fixtures/test_pointer_rewrite.txt";
local validation_data_path="adabrain/resolution/test_fixtures/test_pointer_rewrite.txt";
local device=-1;
local model_size=768;  // tiny: 312     base: 768
local train_nums=35000;
local weight_decay=0.0;
local warmup_steps=547;  // = 35000 / batch_size  64:547  32:1094
local seed=2020;

{
    "dataset_reader": {
        "type": "bert_span_resolution",
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