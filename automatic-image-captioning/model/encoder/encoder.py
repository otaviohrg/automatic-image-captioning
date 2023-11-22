import torch.nn as nn


class EncoderCNN(nn.Module):
    def __init__(self, model, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn
        self.model = model
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.model(images)

        for name, param in self.model.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.require_grad = True
            else:
                param.require_grad = self.train_cnn

        return self.dropout(self.relu(features))
