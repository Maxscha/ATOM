import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class AttentionNet(nn.Module):
    def __init__(self, diff_vectors, msg_vectors):
        super(AttentionNet, self).__init__()
        self.lstm_dim = 256
        self.embedding_diff = nn.Embedding.from_pretrained(diff_vectors, freeze=False)
        self.embedding_msg = nn.Embedding.from_pretrained(msg_vectors, freeze=False)

        self.encoder_diff = nn.LSTM(256, self.lstm_dim, bidirectional=True, batch_first=True)
        self.encoder_msg = nn.LSTM(self.lstm_dim, self.lstm_dim, bidirectional=True, batch_first=True)

        self.diff_attention_layer = Attention(2 * self.lstm_dim, step_dim=200)
        self.msg_attention_layer = Attention(2 * self.lstm_dim, step_dim=15)
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

        diff_lstm_atten = self.diff_attention_layer(diff_lstm)
        msg_lstm_atten = self.msg_attention_layer(msg_lstm)
        combined = torch.cat((diff_lstm_atten, msg_lstm_atten), 1)
        conc = self.relu(self.dropout(self.linear(combined)))
        out = self.out(conc)
        return out
