import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        embedding_mat = torch.from_numpy(np.load(cfg.embed_matrix_fn))
        self.embedding = nn.Embedding.from_pretrained(embedding_mat, freeze=cfg.train_embedding)

        self.lstm_size_in = 300

        self.hidden_size = cfg.lstm_dim
        self.lstm_dropout_rate = cfg.lstm_dropout
        self.bidirectional = cfg.bidirectional
        self.num_layers = cfg.num_layers

        self.lstm = nn.LSTM(input_size=self.lstm_size_in, hidden_size=self.hidden_size, batch_first=True,
                            dropout=self.lstm_dropout_rate, num_layers=self.num_layers,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            lstm_out_size = 2 * self.hidden_size
        else:
            lstm_out_size = self.hidden_size

        self.out = nn.Sequential(nn.Linear(lstm_out_size, 1), nn.Sigmoid())

    def forward(self, text_seq):
        """
        :param text_seq: Incoming text to have the dimensions [B, T], batch_first
        """
        embedded_mat = self.embedding(text_seq)
        h, c = self.lstm(embedded_mat)

        h_in = h[:, -1]

        pred = self.out(h_in)

        return pred.squeeze()
