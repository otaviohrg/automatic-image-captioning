import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .vocabulary import Vocabulary


class ImageDataset(Dataset):
    def __init__(self, data_path, annotation_file, transform=None, freq_threshold=5):
        self.data_path = data_path
        self.df = pd.read_csv(annotation_file)
        self.transform = transform

        self.images = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.images[index]
        image = Image.open(os.path.join(self.data_path, img_id)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        numeric_caption = [self.vocab.string_to_index["<SOS>"]]
        numeric_caption += self.vocab.text_to_numbers(caption)
        numeric_caption.append(self.vocab.string_to_index["<EOS>"])

        return image, torch.tensor(numeric_caption)
