
fs = 16000
chunk_len = 3  # 训练时的 Mixture 长度 (秒)
chunk_size = chunk_len * fs

nnet_conf = {
    "n_srcs": 1,
    "n_fft": 256,             # 16ms 窗长
    "stride": 128,            # 8ms 步长
    "n_layers": 4,            # TFGridNet 层数
    "lstm_hidden_units": 128,
    "emb_dim": 64,
    "emb_ks": 4,
    "emb_hs": 4,
    "sample_rate": fs,
    "use_memonger": True,
    "window": "hann"
}

adam_kwargs = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-7,
    "patience": 5,
    "factor": 0.5,
    "logging_period": 200
}

train_dir = "/node/yym/node/LExt/data/train/"
dev_dir = "/node/yym/node/LExt/data/dev/"

train_data = {
    "mix_scp": train_dir + "mix_clean.scp",
    "ref_scp": train_dir + "ref.scp",  # 目标语音标签
    "aux_scp": train_dir + "auxs1.scp",  # Enrollment 注册语音
    "sample_rate": fs,
}

dev_data = {
    "mix_scp": dev_dir + "mix_clean.scp",
    "ref_scp": dev_dir + "ref.scp",
    "aux_scp": dev_dir + "auxs1.scp",
    "sample_rate": fs,
}
