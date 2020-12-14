// local variables
// reader相关参数
local vocab_path="/home/zs261988/data/vocab/bert_vocabulary";
local max_enc_len=256;
local max_dec_len=30;
local max_turn_len=4;
local do_lower_case=true;
local index_name="tokens";
local padding_token="[PAD]";
local oov_token="[UNK]";
// embedder相关参数
local pretrained_embedding_dim=256;     // 预训练词向量维度
local embedding_size=512;               // 最后映射的维度
local pretrained_file="/home/zs261988/models/ptms/word2vec.txt";  // 预训练词向量文件
local padding_index=0;
// model相关参数
// encoder相关参数
local encoder_num_layers=4;
local feedforward_hidden_dim=2048;
local num_attention_heads=8;
local positional_encoding="sinusoidal";    // encoder位置编码的类型，共享权重时不使用
local encoder_activation="gelu";
local share_encoder_params=false;
// decoder参数
local decoder_type="transformer";
local decoder_num_layers=4;
local use_positional_encoding=false;    // decoder是否使用位置编码
local share_decoder_params=false;
//其他参数
local min_dec_len=4;
local beam_size=2;
local coverage_factor=0.0;
local device=1;
local dropout_rate=0.1;
local seed=2020;
local valid_metric_keys=["bleu_4", "rouge_2_r", "rouge_l_f"];
// 训练相关参数
local l1_reg=1e-5;
local l2_reg=1e-5;
local batch_size=64;
local optimizer_type="huggingface_adamw";
local lr=1e-4;
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
    "type": "pointer_rewrite",
    "vocab_path": vocab_path,
    "max_enc_len": max_enc_len,
    "max_dec_len": max_dec_len,
    "max_turn_len": max_turn_len,
    "index_name": index_name,
    "do_lower_case": do_lower_case,
    "lazy": false
  },
  "model": {
    "type": "transformer_pointer_for_rewrite",
    "encoder": {
      "type": "pytorch_transformer",
      "input_dim": embedding_size,
      "num_layers": encoder_num_layers,
      "feedforward_hidden_dim": feedforward_hidden_dim,
      "num_attention_heads": num_attention_heads,
      "positional_encoding": positional_encoding,
      "positional_embedding_size": embedding_size,
      "activation": encoder_activation
    },
    "decoder": {
      "type": "rewrite_transformer_decoder",
      "decoding_dim": embedding_size,
      "target_embedding_dim": embedding_size,
      "feedforward_hidden_dim": feedforward_hidden_dim,
      "num_layers": decoder_num_layers,
      "num_attention_heads": num_attention_heads,
      "use_positional_encoding": use_positional_encoding
    },
    "decoder_type": decoder_type,
    "decoder_num_layers": decoder_num_layers,
    "text_field_embedder": {
      "type": "basic",
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": pretrained_embedding_dim,
          "projection_dim": embedding_size,
          "pretrained_file": pretrained_file,
          "padding_index": padding_index
        }
      }
    },
    "share_encoder_params": share_encoder_params,
    "share_decoder_params": share_encoder_params,
    "index_name": index_name,
    "beam_size": beam_size,
    "max_turn_len": max_turn_len,
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
      "betas": [0.9, 0.9]
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