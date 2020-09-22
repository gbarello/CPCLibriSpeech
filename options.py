opt = {
    "data_root_path":"./LibriSpeech/train-clean-100/",
    "dev":"cuda:0",
    "dev_list":[0,2],
    "lr_step_rate":10,
    "init_learning_rate":.001,
    "lr_step_factor":0.25,
    "batch_size":176,
    "n_epochs":1,
    "num_workers":24,
    "test_dev":"cuda:0",
    "test_batch_size":64,
    "tsne_spk_frac":.1,
}