
# Text-Classification-of-Product-Titles-pytorch

Classification for products on e-commerce platform according to product names. 

Solution for The 10th National College Students Service Outsourcing Innovation and Entrepreneurship Competition (【A01】2018 网络零售平台商品分类【浪潮】, 第十届中国大学生服务外包创新创业大赛).

# Requirements

* Python 3
* torch==1.4.0
* jieba
* [AlfredWGA/nlputils](www.nlpasdf.com)
* tqdm

# Dataset

Training set can be downloaded from [here](https://drive.google.com/file/d/1SmTU52ibDFEyz8cWraxEbv9xUxeSW1by/view?usp=sharing).  
Pre-trained Chinese word embeddings are obtained from this repo [Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors).  
Word segementation are done by [jieba](https://github.com/fxsjy/jieba).

# Pre-processing

Place all data files in the path specified in `config.py`.

    python preprocess.py 

# Train our models

We implemented TextCNN, BiLSTM and BiGRU for text classification. We use character level (no word segmentation) and word level input.

    python src/train.py --help
    usage: train.py [-h] [-lr LR] [--batch_size BATCH_SIZE]
                    [--model {textcnn,textbilstm,textbigru}] [--mode {char,word}]
                    [-e EPOCH]

    optional arguments:
    -h, --help            show this help message and exit
    -lr LR, --lr LR       Learning rate. Default 1e-3.
    --batch_size BATCH_SIZE
                            Batch size. Default 128.
    --model {textcnn,textbilstm,textbigru}
                            The classification model. Default textcnn.
    --mode {char,word}    Type of the embedding input.
    -e EPOCH, --epoch EPOCH
                            Number of training epoch.

Expected outputs:

    2020-02-06 17:40:47,792 - __main__ - INFO - {'lr': 0.001, 'batch_size': 128, 'model': 'textcnn', 'mode': 'char', 'epoch': 100}
    2020-02-06 17:40:49,145 - __main__ - INFO - Train set size: 400000, valid set size 100000
    2020-02-06 17:40:51,248 - __main__ - INFO - Model modules:
    2020-02-06 17:40:51,250 - __main__ - INFO - 0 -> ('embedding', Embedding(767, 300, padding_idx=0))
    2020-02-06 17:40:51,251 - __main__ - INFO - 1 -> ('conv1d_2', Conv1d(300, 150, kernel_size=(2,), stride=(1,)))
    2020-02-06 17:40:51,254 - __main__ - INFO - 2 -> ('conv1d_3', Conv1d(300, 150, kernel_size=(3,), stride=(1,)))
    2020-02-06 17:40:51,257 - __main__ - INFO - 3 -> ('conv1d_4', Conv1d(300, 150, kernel_size=(4,), stride=(1,)))
    2020-02-06 17:40:51,258 - __main__ - INFO - 4 -> ('relu', ReLU())
    2020-02-06 17:40:51,259 - __main__ - INFO - 5 -> ('fc1', Linear(in_features=450, out_features=256, bias=True))
    2020-02-06 17:40:51,260 - __main__ - INFO - 6 -> ('dropout', Dropout(p=0.5, inplace=False))
    2020-02-06 17:40:51,261 - __main__ - INFO - 7 -> ('fc2', Linear(in_features=256, out_features=1258, bias=True))
    2020-02-06 17:40:51,262 - __main__ - INFO - Total params: 1,074,312
    2020-02-06 17:40:51,264 - __main__ - INFO - Trainable params: 1,074,312
    2020-02-06 17:42:40,828 - __main__ - INFO - Epoch 0, valid loss 1.01316, valid acc 0.7712
