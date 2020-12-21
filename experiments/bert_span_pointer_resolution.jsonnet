// local variables
local max_turn_len=5;
local max_length=256;
local num_epochs=20;
local batch_size=16;
local bert_lr=1e-5;
local attention_lr=1e-4;
local neg_sample_ratio=0.0;
local l1_reg=0.0; // l1正则化系数
local l2_reg=0.0; // l2正则化系数
local bert_path="/data/models/ptms/chinese_roberta_wwm_ext_pytorch/";
// -- chinese_bert_wwm_pytorch
// -- bert_rbt3_pytorch
// -- albert_tiny_489k
// -- albert_void_tiny
local train_data_path="/data/corpus/restoration_200k/mask_train.txt";    // 200k数据集  all_data/restore_train.txt   业务数据集：mask/mask_train.txt  mask/mask_train_expand.txt
local validation_data_path="/data/corpus/restoration_200k/mask_valid.txt"; // 200k数据集  all_data/restore_valid.txt   业务数据集：mask/mask_val.txt
local task_pretrained_file="";
//local task_pretrained_file="/data/models/spdr/restoration_200k_pretrain/best.th";
// 几个预训练模型名称：
// -- bert_wwm_span_pretrain_cls_and_mask_all_data
// -- bert_wwm_span_pretrain_cls_and_mask_bs
// -- bert_rbt3_span_pretrain_cls_and_mask_bs
// -- albert_tiny_span_pretrain_mask_and_cls
// -- albert_tiny_span_pretrain_cls
// -- albert_tiny_span_pretrain_mask
// -- albert_489k_span_pretrain_cls_and_mask_bs
local device=0;
local model_size=768;  // tiny: 312     base: 768
local train_nums=193668;  // 200k_dataset: origin-193668 expand-703278  service data: origin-35000
local weight_decay=0.0;
local warmup_steps=12105;  // = 1 epoch   train_nums / batch_size
local num_gradient_accumulation_steps=1;
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
        "task_pretrained_file": task_pretrained_file,
        "neg_sample_ratio": neg_sample_ratio,
        "max_turn_len": max_turn_len,
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
        "validation_metric": "+semr",
        "num_gradient_accumulation_steps": num_gradient_accumulation_steps
    }
}