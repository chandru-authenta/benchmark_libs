import torch
import webdataset as wds
from torchvision import transforms


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, urls):
        self.urls = urls
        self.to_tensor = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    def __iter__(self):

        urls = self.urls

        # Create WebDataset pipeline
        dataset = wds.WebDataset(
            urls,
            handler=wds.handlers.warn_and_continue,
            shardshuffle=100,  # Enable shard shuffling with buffer size
            empty_check=False,
            cache_size=0,
        )  
        dataset = dataset.decode("pil").to_tuple("png")

        for (image) in dataset:  # unpack single-element tuple
            try:
                image = self.to_tensor(image)  # convert PIL → Tensor
                yield image  # no label since you only have images
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
