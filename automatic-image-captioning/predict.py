import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from data.create_dataloader import get_loader
from model.model import EncoderDecoder
from model.encoder.inception import InceptionEncoder
from model.decoder.lstm import LSTMDecoder
from utils import load_checkpoint, save_checkpoint


def predict():
    with open("config.yml") as config_file:
        config = yaml.safe_load(config_file)

    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    train_loader, dataset = get_loader(
        data_path=config["dataset"]["images_path"],
        annotation_file=config["dataset"]["captions_path"],
        transform=transform,
        num_workers=2,
    )

    #torch.backends.cudnn.benchmark = True
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab) + 1
    num_layers = 1
    num_epochs = 100

    # Initialize
    model = EncoderDecoder(InceptionEncoder, LSTMDecoder, embed_size, hidden_size, vocab_size, num_layers)
    model = model.to(device)

    checkpoint = torch.load(config["file_path"] + "checkpoint.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    index = 48

    image = dataset.__getitem__(index)[0].unsqueeze(0)
    image.to(device)

    print(image.shape)

    #im = Image.open(config["dataset"]["images_path"] + dataset.images[index])
    #im.show()

    print("ORIGINAL: " + dataset.captions[index])
    print("PREDICTED: " + " ".join(model.caption_image(image, dataset.vocab)))


if __name__ == "__main__":
    predict()
