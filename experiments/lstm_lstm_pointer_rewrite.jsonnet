// local variables
// reader相关参数
local vocab_path="/home/zs261988/data/vocab/bert_vocabulary";
local max_enc_len=256;
local max_dec_len=30;
local max_turn_len=3;
local do_lower_case=true;
local index_name="tokens";
local padding_token="[PAD]";
local oov_token="[UNK]";
// embedder相关参数
local pretrained_embedding_dim=256;
local embedding_size=512;
local pretrained_file="/home/zs261988/models/ptms/word2vec.txt";
local padding_index=0;
// model相关参数
local encoder_hidden_size=256;
local encoder_num_layers=4;
local decoder_type="lstm";
local decoder_num_layers=4;
local min_dec_len=4;
local beam_size=4;
local coverage_factor=0.0;
local device=0;
local dropout_rate=0.1;
local seed=2020;
local valid_metric_keys=["bleu_1", "bleu_2", "rouge_1_r", "rouge_2_r", "rouge_l_f"];
// 训练相关参数
local l1_reg=1e-5;
local l2_reg=1e-5;
local batch_size=64;
local optimizer_type="adamw";
local lr=0.001;
local num_epochs=20;


{
  "train_data_path": "/home/zs261988/data/rewrite/mask/mask_train_expand.txt",        // 训练集数据
  "validation_data_path": "/home/zs261988/data/rewrite/mask/mask_val.txt",    // 测试集数据
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
    "type": "lstm_pointer_for_rewrite",
    "embedding_size": embedding_size,
    "encoder_hidden_size": encoder_hidden_size,
    "encoder_num_layers": encoder_num_layers,
    "decoder": {
      "type": "stacked_lstm_decoder",
      "decoding_dim": encoder_hidden_size,
      "target_embedding_dim": embedding_size,
      "num_layers": decoder_num_layers
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
    "index_name": index_name,
    "beam_size": beam_size,
    "max_turn_len": max_turn_len,
    "min_dec_len": min_dec_len,
    "max_dec_len": max_dec_len,
    "coverage_factor": coverage_factor,
    "device": device,
    "valid_metric_keys": valid_metric_keys,
    "dropout_rate": dropout_rate,
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