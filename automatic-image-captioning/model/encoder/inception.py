import torch.nn as nn
import torchvision.models as models

from .abstractEncoder import EncoderCNN


class InceptionEncoder(EncoderCNN):
    def __init__(self, embed_size, train_cnn=False):
        inception = models.inception_v3(pretrained=True, aux_logits=True)
        inception.fc = nn.Linear(inception.fc.in_features, embed_size)
        super(InceptionEncoder, self).__init__(model=inception, train_cnn=train_cnn)

    def forward(self, images):
        features = self.model(images)

        for name, param in self.model.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.require_grad = True
            else:
                param.require_grad = self.train_cnn

        return self.dropout(self.relu(features[0]))
