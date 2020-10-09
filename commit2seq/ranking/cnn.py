import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, diff_vectors, msg_vectors, msg_len, diff_len):
        super().__init__()
        self.kernel_count = 16
        self.kernel_size = [3, 3]
        self.padding = int((self.kernel_size[0] - 1) / 2)
        self.embedding_diff = nn.Embedding.from_pretrained(diff_vectors, freeze=False)
        self.embedding_msg = nn.Embedding.from_pretrained(msg_vectors, freeze=False)
        self.conv_module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.kernel_count, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = Flatten().cuda()
        self.dense_input_size = self.kernel_count * int(msg_len / 2) * int(diff_len / 2)
        self.dense1 = nn.Linear(self.dense_input_size, 1024)
        self.dense2 = nn.Linear(1024, 32)
        self.dense3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, retrieved_msgs, used_tokens):
        embed_msgs = self.embedding_msg(retrieved_msgs).cuda()
        embed_diffs = self.embedding_diff(used_tokens).cuda()
        interaction_matrix = torch.bmm(embed_msgs, embed_diffs.view(embed_diffs.size()[0], embed_diffs.size()[2], embed_diffs.size()[1]))
        interaction_matrix = torch.unsqueeze(interaction_matrix, 1)
        code_cnn_out = self.conv_module(interaction_matrix)
        flatten = self.flatten(code_cnn_out)
        dropout_res = self.dropout(flatten)
        logits = self.relu(self.dense1(dropout_res))
        logits = self.relu(self.dense2(logits))
        logits = self.dense3(logits)
        return logits


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)