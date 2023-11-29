import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from data.create_dataloader import get_loader
from model.model import EncoderDecoder
from model.encoder.inception import InceptionEncoder
from model.decoder.lstm import LSTMDecoder
from utils import load_checkpoint, save_checkpoint
from predict import predict


def train():
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

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    load_model = True
    save_model = True

    #Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab) + 1
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    #Tensorboard
    writer = SummaryWriter()
    step = 0

    #Initialize
    model = EncoderDecoder(InceptionEncoder, LSTMDecoder, embed_size, hidden_size, vocab_size, num_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.string_to_index["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load(config["file_path"] + "checkpoint.pth.tar"), model, optimizer)

    for epoch in range(num_epochs):
        running_loss = 0.
        last_loss = 0.

        predict()

        print("EPOCH: {}".format(step//(len(dataset)//32)))

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for index, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
            running_loss += loss.item()
            if index % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                print('  batch {} loss: {}'.format(index + 1, last_loss))
                running_loss = 0.


if __name__ == "__main__":
    train()
