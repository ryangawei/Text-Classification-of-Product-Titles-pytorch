# coding=utf-8
import torch.nn as nn
import torch
import config
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, mode='char'):
        super(TextCNN, self).__init__()
        if mode == 'char':
            embedding_weights = np.load(config.FIT_CHAR_PATH)['embeddings'].astype(np.float32)
            vocab_size, vec_dim = embedding_weights.shape
            text_length = config.MAX_CHAR_TEXT_LENGTH
        else:
            embedding_weights = np.load(config.FIT_WORD_PATH)['embeddings'].astype(np.float32)
            vocab_size, vec_dim = embedding_weights.shape
            text_length = config.MAX_WORD_TEXT_LENGTH

        embedding_weights = torch.from_numpy(embedding_weights)
        self.embedding = nn.Embedding.from_pretrained(embedding_weights,
                                                      freeze=False,
                                                      padding_idx=0)

        self.num_classes = config.NUM_CLASSES
        self.filter_sizes = [2, 3, 4]
        self.num_filter = 150

        self.conv1d_layers = []
        for size in self.filter_sizes:
            conv1d = nn.Conv1d(vec_dim, self.num_filter, kernel_size=size)
            nn.init.xavier_normal_(conv1d.weight.data)
            setattr(self, 'conv1d_{}'.format(size), conv1d)
            self.conv1d_layers.append(conv1d)

        self.relu = nn.ReLU()
        # self.global_maxpool = lambda x: torch.max(x, dim=2)

        self.fc1 = nn.Linear(self.num_filter * len(self.filter_sizes), 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, self.num_classes)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        # (batch, vec_dim, length)
        x = x.permute(0, 2, 1)

        conv1d_outputs = []
        for conv1d in self.conv1d_layers:
            o = self.relu(conv1d(x))
            # Global max pooling
            # (batch, num_filter)
            o, _ = o.max(dim=2)
            conv1d_outputs.append(o)

        x = torch.cat(conv1d_outputs, dim=1)

        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x


class TextBiLSTM(nn.Module):
    def __init__(self, mode='char'):
        super(TextBiLSTM, self).__init__()
        if mode == 'char':
            embedding_weights = np.load(config.FIT_CHAR_PATH)['embeddings'].astype(np.float32)
            vocab_size, vec_dim = embedding_weights.shape
            text_length = config.MAX_CHAR_TEXT_LENGTH
        else:
            embedding_weights = np.load(config.FIT_WORD_PATH)['embeddings'].astype(np.float32)
            vocab_size, vec_dim = embedding_weights.shape
            text_length = config.MAX_WORD_TEXT_LENGTH

        embedding_weights = torch.from_numpy(embedding_weights)
        self.embedding = nn.Embedding.from_pretrained(embedding_weights,
                                                      freeze=False,
                                                      padding_idx=0)

        self.hidden_size = 64
        self.bilstm = nn.LSTM(input_size=vec_dim, 
                            hidden_size=self.hidden_size, 
                            num_layers=2, 
                            batch_first=True,
                            dropout=0.7,
                            bidirectional=True)

        self.num_classes = config.NUM_CLASSES  
        self.fc = nn.Linear(self.hidden_size*2, self.num_classes)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        # Only use the final step.
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class TextBiGRU(nn.Module):
    def __init__(self, mode='char'):
        super(TextBiGRU, self).__init__()
        if mode == 'char':
            embedding_weights = np.load(config.FIT_CHAR_PATH)['embeddings'].astype(np.float32)
            vocab_size, vec_dim = embedding_weights.shape
            text_length = config.MAX_CHAR_TEXT_LENGTH
        else:
            embedding_weights = np.load(config.FIT_WORD_PATH)['embeddings'].astype(np.float32)
            vocab_size, vec_dim = embedding_weights.shape
            text_length = config.MAX_WORD_TEXT_LENGTH

        embedding_weights = torch.from_numpy(embedding_weights)
        self.embedding = nn.Embedding.from_pretrained(embedding_weights,
                                                      freeze=False,
                                                      padding_idx=0)

        self.hidden_size = 64
        self.gru = nn.GRU(input_size=vec_dim, 
                            hidden_size=self.hidden_size, 
                            num_layers=2, 
                            batch_first=True,
                            dropout=0.7,
                            bidirectional=True)

        self.num_classes = config.NUM_CLASSES  
        self.fc = nn.Linear(self.hidden_size*2, self.num_classes)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x, _ = self.gru(x)
        # Only use the final step.
        x = x[:, -1, :]
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = TextBiLSTM()
    x = torch.ones(size=[128, 25], dtype=torch.long)
    output = model(x)
    print(output.shape)



