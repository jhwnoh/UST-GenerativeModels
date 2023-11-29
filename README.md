# UST-GenerativeModels

[Only for practice purpose]


This is repository for practice code for UST2023-Fall 'Materials Informatics'

- Model_GRU/RNN.pth.tar : Pretrained model for Autoregressive GM




## Try with GPU to train model


### TryWithGPU_Autoregressive : training autoregressive GM with GPU
[Need to download train/test set from MOSES repository]

1. Model training (chkpt file is saved in CharGRU directory)
```
python gru_traincode_gpu.py
```

2. Sampling (saved into .pkl file)
```
python gru_sampling_gpu.py [saved model file] [name to save sampled results]
```
[Example] if name of the saved model file is Model_GRU.pth.tar, and if we want to save the sampled result in random_sampled.pkl file,
```
python gru_sampling_gpu.py Model_GRU.pth.tar random_sampled.pkl
```
