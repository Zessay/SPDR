// local variables
// reader相关参数
local model_name="/home/zs261988/models/ptms/albert_void_tiny";   // 预训练模型的路径
local vocab_path="/home/zs261988/data/vocab/bert_vocabulary";
local max_enc_len=256;
local max_dec_len=30;
local max_turn_len=4;
local do_lower_case=true;
local index_name="bert";
local padding_token="[PAD]";
local oov_token="[UNK]";
// model相关参数
// encoder参数
local encoder_embedding_size=312;   // albert_tiny为312，bert_base为768
// decoder参数
local decoder_type="transformer";
local decoder_num_layers=4;
local feedforward_hidden_dim=1248;    // albert_tiny的中间FFN层的参数为1248
local num_attention_heads=8;
local use_positional_encoding=false;
local share_decoder_params=true;      // 是否共享decoder参数
// 其他参数
local min_dec_len=4;
local beam_size=4;
local coverage_factor=0.0;
local device=0;
local dropout_rate=0.1;
local seed=2020;
local valid_metric_keys=["bleu_4", "rouge_2_r", "rouge_l_f"];
// 训练相关参数
local l1_reg=1e-5;
local l2_reg=1e-5;
local batch_size=64;
local optimizer_type="huggingface_adamw";
local lr=3e-5;
local num_epochs=20;

{
  "train_data_path": "/home/zs261988/data/rewrite/all_data/all_train.txt",        // 训练集数据
  "validation_data_path": "/home/zs261988/data/rewrite/all_data/all_val.txt",    // 测试集数据
  // 根据bert词表转化得到的可以用Vocabulary加载的词表
  "vocabulary": {
    "type": "from_files",
    "directory": vocab_path,
    "padding_token": padding_token,
    "oov_token": oov_token,
  },
  "dataset_reader": {
    "type": "bert_pointer_rewrite",
    "model_name": model_name,
    "vocab_path": vocab_path,
    "max_enc_len": max_enc_len,
    "max_dec_len": max_dec_len,
    "max_turn_len": max_turn_len,
    "index_name": index_name,
    "pad_token": padding_token,
    "oov_token": oov_token,
    "do_lower_case": do_lower_case,
    "lazy": false
  },
  "model": {
    "type": "bert_pointer_for_rewrite",
    "model_name": model_name,
    "decoder": {
      "type": "rewrite_transformer_decoder",
      "decoding_dim": encoder_embedding_size,
      "target_embedding_dim": encoder_embedding_size,
      "feedforward_hidden_dim": feedforward_hidden_dim,
      "num_layers": 1,
      "num_attention_heads": num_attention_heads,
      "use_positional_encoding": use_positional_encoding
    },
    "decoder_type": decoder_type,
    "decoder_num_layers": decoder_num_layers,
    "share_decoder_params": share_decoder_params,
    "index_name": index_name,
    "beam_size": beam_size,
    "min_dec_len": min_dec_len,
    "max_dec_len": max_dec_len,
    "coverage_factor": coverage_factor,
    "device": device,
    "valid_metric_keys": valid_metric_keys,
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
    },
    "regularizer": {
      "regexes": [
        [".*weight",
          {
            "type": "l2",
            "alpha": l2_reg
        }],
        [".*weight",
          {
            "type": "l1",
            "alpha": l1_reg
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
      "type": optimizer_type,
      "lr": lr,
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