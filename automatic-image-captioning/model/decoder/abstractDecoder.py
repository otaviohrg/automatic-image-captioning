import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, model, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.model = model
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.vocab_size = vocab_size

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hidden, _ = self.model(embeddings)
        outputs = self.linear(hidden)
        return outputs


