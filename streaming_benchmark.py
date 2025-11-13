
import csv
import io
import os
import time
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from streaming import StreamingDataset
from PIL import Image

# Custom collate function to handle PIL images
def collate_fn(batch):
    """Collate function that converts PIL images to tensors."""
    to_tensor = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    images = []
    labels = []
    
    for sample in batch:
        img = sample.get('image')
        label = sample.get('label')
        
        # Convert PIL image to tensor if needed
        if isinstance(img, Image.Image):
            img = to_tensor(img)
        elif isinstance(img, bytes):
            # If image is bytes, load as PIL then convert
            img = Image.open(io.BytesIO(img))
            img = to_tensor(img)
        
        images.append(img)
        labels.append(label)
    
    # Stack images and labels
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return {'image': images, 'label': labels}

def run_single_benchmark(dataset, local_cache_path, run_id):
    """Run a single benchmark iteration and return metrics.

    Accepts a StreamingDataset instance (or any PyTorch dataset) and a local
    cache path to use for disk-usage metrics.
    """
    print(f"\n🔄 Running benchmark iteration {run_id + 1}...")

    # # Init GPU monitoring
    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU:0
    process = psutil.Process(os.getpid())

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print("🔎 Measuring dataset loading performance...")
    all_data_load_times = []
    all_cpu, all_ram, all_disk = [], [], []
    total_images = 0

    def get_metrics():
        cpu = psutil.cpu_percent(interval=None)
        ram = process.memory_info().rss / (1024**3)
        # Use the provided local cache path (cross-platform) to measure dataset disk usage
        try:
            disk = psutil.disk_usage(local_cache_path).used / (1024**3)
        except Exception:
            # Fall back to root if the provided path is invalid
            disk = psutil.disk_usage("/").used / (1024**3)
        # util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        # gpu_util = util.gpu
        # gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)
        return cpu, ram, disk

    def _batch_length(batch):
        """Return the number of samples in a dataloader batch.

        Supports common batch shapes: dicts of tensors/lists, lists/tuples,
        single tensors, and fallbacks.
        """
        if isinstance(batch, dict):
            # Get first value and infer its length
            try:
                val = next(iter(batch.values()))
            except StopIteration:
                return 0
            # Tensors and sequences
            try:
                return len(val)
            except Exception:
                try:
                    return val.size(0)
                except Exception:
                    return 1
        if isinstance(batch, (list, tuple)):
            return len(batch)
        if hasattr(batch, "size"):
            try:
                return batch.size(0)
            except Exception:
                return 1
        # Fallback
        try:
            return len(batch)
        except Exception:
            return 1

    start_time = time.time()
    dataloader_iter = iter(dataloader)
    batch_idx = 0

    while True:
        try:
            
            batch_start = time.time()
            batch = next(dataloader_iter)
            batch_n = _batch_length(batch)
            print(f"Processed batch of {batch_n} images")
            data_load_time = time.time() - batch_start
            all_data_load_times.append(data_load_time)

            batch_idx += 1
        except StopIteration:
            break

        cpu, ram, disk = get_metrics()
        all_cpu.append(cpu)
        all_ram.append(ram)
        all_disk.append(disk)
        # all_gpu_util.append(gpu_util)
        # all_gpu_mem.append(gpu_mem)

        total_images += batch_n

    total_time = time.time() - start_time
    avg_data_load_time = (
        sum(all_data_load_times) / len(all_data_load_times)
        if all_data_load_times
        else 0
    )
    avg_cpu = sum(all_cpu) / len(all_cpu) if all_cpu else 0
    avg_ram = sum(all_ram) / len(all_ram) if all_ram else 0
    # avg_gpu_util = sum(all_gpu_util) / len(all_gpu_util) if all_gpu_util else 0
    images_per_sec = total_images / total_time if total_time > 0 else 0

    # Return metrics for this run
    return {
        "total_time": total_time,
        "total_images": total_images,
        "images_per_sec": images_per_sec,
        "avg_data_load_time": avg_data_load_time,
        "avg_cpu": avg_cpu,
        "avg_ram": avg_ram,
        # "avg_gpu_util": avg_gpu_util,
    }



def main():

    local = 's3://authenta-streaming-data/mds_shards/'
    batch_size = 32  # Must match DataLoader batch_size

    print(f"Loading dataset from: {local}")

    dataset = StreamingDataset(remote=local, local='./cache', batch_size=batch_size, shuffle=True)

    # Number of benchmark runs
    num_runs = 1
    print(f"\n🚀 Starting {num_runs} benchmark iterations...")

    # Collect metrics from all runs
    all_metrics = []

    for run_id in range(num_runs):
        metrics = run_single_benchmark(dataset, local, run_id)
        all_metrics.append(metrics)

        # Print individual run results
        print(f"\n📊 Run {run_id + 1} Results:")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Total images: {metrics['total_images']}")
        print(f"  Images per second: {metrics['images_per_sec']:.2f}")
        print(f"  Avg data load time: {metrics['avg_data_load_time']:.4f}s")

    # Calculate statistics across all runs
    metrics_keys = [
        "total_time",
        "total_images",
        "images_per_sec",
        "avg_data_load_time",
        "avg_cpu",
        "avg_ram",
        # "avg_gpu_util",
    ]

    stats = {}
    for key in metrics_keys:
        values = [run[key] for run in all_metrics]
        stats[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }

    # Save detailed metrics to CSV
    metrics_file = "metrics/streaming_metrics_s3_images.csv"
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
    print("📈 BENCHMARK SUMMARY STATISTICS (5 runs)")
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

    # print("\n🎮 GPU Utilization:")
    # mean_gpu = stats["avg_gpu_util"]["mean"]
    # std_gpu = stats["avg_gpu_util"]["std"]
    # print(f"  Mean: {mean_gpu:.1f}% ± {std_gpu:.1f}%")

    print(f"\n💾 Results saved to: {metrics_file}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
