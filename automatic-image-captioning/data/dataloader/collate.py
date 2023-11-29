import torch
from torch.nn.utils.rnn import pad_sequence


class Collate:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_index)

        return images, captions
    
