import csv
import os
import time
import numpy as np
import psutil
from torchvision import transforms
import torch
import ray
from typing import Any, Dict

def transform_image(row: Dict[str, Any]) -> Dict[str, Any]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])
    row["image"] = transform(row["image"])
    return row


def benchmark_ray_direct_loading(local, run_id, batch_size=32, num_workers=4):
    print(f"\n🔄 Running benchmark iteration {run_id + 1}...")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers, ignore_reinit_error=True)
    
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
    
    ds = (
        ray.data.read_images(local)
        .map(transform_image)
    )

    start_time = time.time()
    batch_idx = 0
    batch_time=time.time()

    # Iterate over batches
    for batch in ds.iter_torch_batches(batch_size=batch_size, local_shuffle_buffer_size=250):
        elapsed=time.time()-batch_time
        
        all_data_load_times.append(time.time() - elapsed)
        
        total_images += len(batch["image"])
        batch_idx += 1
        batch_time=time.time()
        
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
        metrics = benchmark_ray_direct_loading(local, run_id)
        all_metrics.append(metrics)

        # Print individual run results
        print(f"\n📊 Run {run_id + 1} Results:")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Total images: {metrics['total_images']}")
        print(f"  Images per second: {metrics['images_per_sec']:.2f}")
        print(f"  Avg data load time: {metrics['avg_data_load_time']:.4f}s")
        
        ray.shutdown()
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
    metrics_file = "metrics/ray-data_metrics_10000_images.csv"
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
    
    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()