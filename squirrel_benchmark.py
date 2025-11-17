import csv
import os
import time
import numpy as np
import psutil
from torchvision import transforms
import torch
from squirrel.iterstream.source import IterableSource
from torch.utils.data import IterableDataset, DataLoader
import torch.utils.data
from PIL import Image
from pathlib import Path

class ImageDataset(IterableDataset):
    """
    IterableDataset with proper worker splitting.
    Each worker processes different data shards.
    Applies 512x512 resize transform.
    """
    
    def __init__(self, folder_path, recursive=True, transform=None):
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



def benchmark_squirrel_direct_loading(local, run_id):
    print(f"\n🔄 Running benchmark iteration {run_id + 1}...")
    process = psutil.Process(os.getpid())

    print("🔎 Measuring dataset loading performance...")
    all_data_load_times = []
    all_cpu, all_ram, all_disk = [], [], []
    total_images = 0

    def get_metrics():
        cpu = psutil.cpu_percent(interval=None)
        ram = process.memory_info().rss / (1024**3)
        try:
            disk = psutil.disk_usage(local).used / (1024**3)
        except Exception:
            disk = psutil.disk_usage("/").used / (1024**3)
        return cpu, ram, disk
    
    dataset = ImageDataset(local, recursive=True)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        prefetch_factor=2
    )


    start_time = time.time()
    batch_idx = 0
    first_batch = True
    for batch in dataloader:
        batch_start = time.time()
        data_load_time = time.time() - batch_start
        all_data_load_times.append(data_load_time)
        batch_idx += 1
        batch_size_actual = len(batch['path'])
        total_images += batch_size_actual
        
        # Print info about first batch
        if first_batch:
            image_shape = batch['image'].shape
            print(f"✓ First batch - Tensor shape: {image_shape}")
            print()
            first_batch = False
        print(f"process batch of {len(batch['image'])} images")
        
        # Collect metrics
        cpu, ram, disk = get_metrics()
        all_disk.append(disk)
        all_cpu.append(cpu)
        all_ram.append(ram)
    
    total_time = time.time() - start_time
    avg_data_load_time = np.mean(all_data_load_times) if all_data_load_times else 0
    avg_cpu = np.mean(all_cpu) if all_cpu else 0
    avg_ram = np.mean(all_ram) if all_ram else 0
    images_per_sec = total_images / total_time if total_time > 0 else 0

    # Return metrics for this run
    return {
        "total_time": total_time,
        "total_images": total_images,
        "images_per_sec": images_per_sec,
        "avg_data_load_time": avg_data_load_time,
        "avg_cpu": avg_cpu,
        "avg_ram": avg_ram,
    }
        

def main():
    local = '/home/ubuntu/s3mount/original_dataset'

    print(f"Loading dataset from: {local}")


    # Number of benchmark runs
    num_runs = 1
    print(f"\n🚀 Starting {num_runs} benchmark iterations...")

    # Collect metrics from all runs
    all_metrics = []

    for run_id in range(num_runs):
        metrics = benchmark_squirrel_direct_loading(local, run_id)
        all_metrics.append(metrics)

        # Print individual run results
        print(f"\n📊 Run {run_id + 1} Results:")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Total images: {metrics['total_images']}")
        print(f"  Images per second: {metrics['images_per_sec']:.2f}")
        print(f"  Avg data load time: {metrics['avg_data_load_time']:.4f}s")
        

        time.sleep(2)

    # Calculate statistics across all runs
    metrics_keys = [
        "total_time",
        "total_images",
        "images_per_sec",
        "avg_data_load_time",
        "avg_cpu",
        "avg_ram",
    ]

    stats = {}
    for key in metrics_keys:
        values = [run[key] for run in all_metrics]
        stats[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }

    # Save detailed metrics to CSV
    metrics_file = "ec2_metrics/squirrel-data_metrics_10000_images.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header for individual runs
        writer.writerow(["run_id"] + metrics_keys)

        # Write individual run data
        for i, metrics in enumerate(all_metrics):
            writer.writerow([i + 1] + [metrics[key] for key in metrics_keys])

        # Write empty line and statistics
        writer.writerow([])
        writer.writerow(["statistic"] + metrics_keys)
        writer.writerow(["mean"] + [stats[key]["mean"] for key in metrics_keys])
        writer.writerow(["std"] + [stats[key]["std"] for key in metrics_keys])

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print(f"📈 BENCHMARK SUMMARY STATISTICS ({num_runs} runs)")
    print("=" * 60)

    print("\n🕐 Total Time:")
    mean_time = stats["total_time"]["mean"]
    std_time = stats["total_time"]["std"]
    print(f"  Mean: {mean_time:.2f}s ± {std_time:.2f}s")

    print("\n🚀 Images per Second:")
    mean_ips = stats["images_per_sec"]["mean"]
    std_ips = stats["images_per_sec"]["std"]
    print(f"  Mean: {mean_ips:.2f} ± {std_ips:.2f}")

    print("\n⚡ Average Data Load Time:")
    mean_load = stats["avg_data_load_time"]["mean"]
    std_load = stats["avg_data_load_time"]["std"]
    print(f"  Mean: {mean_load:.4f}s ± {std_load:.4f}s")

    print("\n💾 CPU Usage:")
    mean_cpu = stats["avg_cpu"]["mean"]
    std_cpu = stats["avg_cpu"]["std"]
    print(f"  Mean: {mean_cpu:.1f}% ± {std_cpu:.1f}%")

    print("\n🧠 RAM Usage:")
    mean_ram = stats["avg_ram"]["mean"]
    std_ram = stats["avg_ram"]["std"]
    print(f"  Mean: {mean_ram:.2f}GB ± {std_ram:.2f}GB")

    print(f"\n💾 Results saved to: {metrics_file}")
    


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()

