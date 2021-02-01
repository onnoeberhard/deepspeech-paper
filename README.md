# deepspeech-paper
This repository contains additional resources for the paper "Effects of Layer Freezing when Transferring DeepSpeech to New Languages".

The steps taken for training the models are compiled in the file `training.sh`. The training and testing output logs can be found in the `logs` directory. The paper itself (LaTeX and pdf) is in the `paper` directory, the plots of the learning curves (ipynb and pdf) are in the root directory.

The modified versions of DeepSpeech that utilize layer freezing can be found at https://github.com/onnoeberhard/deepspeech-transfer, the different versions with a different number of frozen layers are in the branches _transfer-1_, _transfer-2_, _transfer_ and _transfer-4_.
