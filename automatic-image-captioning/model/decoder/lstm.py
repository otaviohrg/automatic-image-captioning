import torch.nn as nn
from decoder import DecoderRNN


class LSTMDecoder(DecoderRNN):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        super(LSTMDecoder, self).__init__(embed_size, lstm, hidden_size, vocab_size)
