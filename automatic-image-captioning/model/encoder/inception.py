import torch.nn as nn
import torchvision.models as models

from encoder import EncoderCNN


class InceptionEncoder(EncoderCNN):
    def __init__(self, embed_size, train_cnn=False):
        inception = models.inception_v3(pretrained=True, aux_logits=False)
        inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        super(InceptionEncoder, self).__init__(model=self.inception, train_cnn=train_cnn)

