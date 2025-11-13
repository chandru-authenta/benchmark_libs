"""
Squirrel Core with PyTorch DataLoader - Image Counter with Worker Splitting
"""

from squirrel.iterstream.source import IterableSource
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch.utils.data


class ImageDataset(IterableDataset):
    """
    IterableDataset with proper worker splitting.
    Each worker processes different data shards.
    Applies 512x512 resize transform.
    """
    
    def __init__(self, folder_path, recursive=False, transform=None):
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        if recursive:
            self.image_files = [
                str(f) for f in sorted(Path(folder_path).rglob('*'))
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        else:
            self.image_files = [
                str(f) for f in sorted(Path(folder_path).glob('*'))
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        
        # Default transform: resize to 512x512 and convert to tensor
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        print(f"✓ Found {len(self.image_files)} images")
    
    def __iter__(self):
        """Stream with worker splitting - each worker gets different data"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # No workers, process all
            start_idx = 0
            num_workers = 1
        else:
            # Multiple workers - split data
            start_idx = worker_info.id
            num_workers = worker_info.num_workers
        
        # Each worker gets every nth item
        worker_files = [
            self.image_files[i] 
            for i in range(start_idx, len(self.image_files), num_workers)
        ]
        
        source = IterableSource(worker_files)
        
        for image_path in source:
            try:
                # Load and transform image
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.transform(image)
                
                yield {
                    'image': image_tensor,
                    'path': image_path,
                    'filename': Path(image_path).name
                }
            except Exception as e:
                print(f"Warning: Failed to load {image_path}: {e}")
                continue


def stream_images(folder_path, batch_size=32, recursive=False, num_workers=4):
    print("=" * 70)
    print("Squirrel Core + PyTorch DataLoader - Image Counter")
    print("=" * 70)
    print(f"Folder: {folder_path}")
    print(f"Recursive: {recursive}")
    print(f"Num Workers: {num_workers}")
    print(f"Batch Size: {batch_size}")
    print(f"Transform: Resize to 512x512 + ToTensor")
    print("=" * 70)
    print()
    
    dataset = ImageDataset(folder_path, recursive=recursive)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2
    )
    
    image_count = 0
    batch_count = 0
    first_batch = True
    
    for batch in dataloader:
        batch_count += 1
        batch_size_actual = len(batch['path'])
        image_count += batch_size_actual
        
        # Print info about first batch
        if first_batch:
            image_shape = batch['image'].shape
            print(f"✓ First batch - Tensor shape: {image_shape}")
            print()
            first_batch = False
        
        print(f"Batch {batch_count}: {batch_size_actual} images | Total: {image_count}")
    
    print()
    print("=" * 70)
    print(f"✓ Total batches: {batch_count}")
    print(f"✓ Total images processed: {image_count}")
    print("=" * 70)
    return image_count


if __name__ == "__main__":
    folder_path = "./original_dataset/dragon_train_000"
    stream_images(folder_path, batch_size=32, recursive=False, num_workers=4)