import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, diff_vectors, msg_vectors):
        super(RNN, self).__init__()
        self.lstm_dim = 256
        self.embedding_diff = nn.Embedding.from_pretrained(diff_vectors, freeze=False)
        self.embedding_msg = nn.Embedding.from_pretrained(msg_vectors, freeze=False)
        self.encoder_diff = nn.LSTM(256, self.lstm_dim, bidirectional=True, batch_first=True)
        self.encoder_msg = nn.LSTM(self.lstm_dim, self.lstm_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(4 * self.lstm_dim, self.lstm_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.lstm_dim, 1)

    def forward(self, retrieved_msgs, used_tokens):
        diff_embedding = self.embedding_diff(used_tokens).cuda()
        diff_embedding = torch.squeeze(torch.unsqueeze(diff_embedding, 0))
        msg_embedding = self.embedding_msg(retrieved_msgs).cuda()
        msg_embedding = torch.squeeze(torch.unsqueeze(msg_embedding, 0))
        diff_lstm, _ = self.encoder_diff(diff_embedding)
        msg_lstm, _ = self.encoder_msg(msg_embedding)
        diff_lstm = diff_lstm[:, -1, :]
        msg_lstm = msg_lstm[:, -1, :]
        combined = torch.cat((diff_lstm, msg_lstm), 1)
        conc = self.relu(self.dropout(self.linear(combined)))
        out = self.out(conc)
        return out
