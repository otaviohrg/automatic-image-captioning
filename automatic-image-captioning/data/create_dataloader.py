import yaml
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.dataloader.dataset import ImageDataset
from data.dataloader.collate import Collate


def get_loader(
        data_path,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
):
    dataset = ImageDataset(
        data_path=data_path,
        annotation_file=annotation_file,
        transform=transform)

    pad_index = dataset.vocab.string_to_index["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_index=pad_index)
    )

    return loader, dataset


if __name__ == "__main__":
    with open("../config.yml") as config_file:
        config = yaml.safe_load(config_file)

    testTransform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), ]
    )

    testLoader, testDataset = get_loader(
        data_path=config["dataset"]["images_path"],
        annotation_file=config["dataset"]["captions_path"],
        transform=testTransform,
    )
    
    for idx, (images, captions) in enumerate(testLoader):
        print(images.shape)
        print(captions.shape)
