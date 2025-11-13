import csv
import os
import time
import numpy as np
import psutil
from torchvision import transforms
import torch
import ray
from typing import Any, Dict
import pyarrow.fs  # ✅ needed for custom S3 filesystem

# Optional: globally disable checksum validation
os.environ["ARROW_S3_CHECKSUM"] = "0"
os.environ["AWS_S3_DISABLE_CHECKSUM_VALIDATION"] = "true"
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

def transform_image(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert PIL image to tensor and resize."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])
    row["image"] = transform(row["image"])
    return row


def benchmark_ray_direct_loading(s3_path, cache_path, run_id, batch_size=32, num_workers=4):
    print(f"\n🔄 Running benchmark iteration {run_id + 1}...")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=num_workers,
            ignore_reinit_error=True,
            runtime_env={
                "env_vars": {
                    "AWS_S3_DISABLE_CHECKSUM_VALIDATION": "true",
                    "PYARROW_IGNORE_TIMEZONE": "1",
                    "AWS_RETRY_ATTEMPTS": "3",
                    "ARROW_S3_CHECKSUM": "0",
                }
            },
        )

    process = psutil.Process(os.getpid())

    print("🔎 Measuring dataset loading performance...")
    all_data_load_times = []
    all_cpu, all_ram, all_disk = [], [], []
    total_images = 0

    def get_metrics():
        """Collect CPU, RAM, and disk usage metrics."""
        cpu = psutil.cpu_percent(interval=None)
        ram = process.memory_info().rss / (1024**3)
        try:
            disk = psutil.disk_usage(cache_path).used / (1024**3)
        except Exception:
            disk = psutil.disk_usage("/").used / (1024**3)
        return cpu, ram, disk

    # ✅ Create a custom S3 filesystem with checksum validation disabled
    try:
        s3_fs = pyarrow.fs.S3FileSystem(
            anonymous=False,
            region="us-east-1",              # change if your bucket is in a different region
            connect_timeout=60,
            allow_bucket_creation=False,
            background_writes=False,
            validate_object_checksum=False,  # 🚫 disables checksum validation
        )

        ds = (
            ray.data.read_images(
                s3_path,
                filesystem=s3_fs,
            )
            .map(transform_image)
        )
        print("✅ Successfully connected to S3 and loaded dataset with checksum disabled.")

    except Exception as e:
        print(f"❌ Error initializing S3 filesystem: {e}")
        print("⚠️ Retrying without custom filesystem...")
        ds = ray.data.read_images(s3_path).map(transform_image)

    start_time = time.time()
    batch_idx = 0

    # Iterate over batches
    for batch in ds.iter_torch_batches(batch_size=batch_size, local_shuffle_buffer_size=250):
        batch_start = time.time()
        data_load_time = time.time() - batch_start
        all_data_load_times.append(data_load_time)

        total_images += len(batch["image"])
        batch_idx += 1
        print(f"🖼️ Processed batch of {len(batch['image'])} images")

        cpu, ram, disk = get_metrics()
        all_cpu.append(cpu)
        all_ram.append(ram)
        all_disk.append(disk)

    total_time = time.time() - start_time
    avg_data_load_time = np.mean(all_data_load_times) if all_data_load_times else 0
    avg_cpu = np.mean(all_cpu) if all_cpu else 0
    avg_ram = np.mean(all_ram) if all_ram else 0
    images_per_sec = total_images / total_time if total_time > 0 else 0

    return {
        "total_time": total_time,
        "total_images": total_images,
        "images_per_sec": images_per_sec,
        "avg_data_load_time": avg_data_load_time,
        "avg_cpu": avg_cpu,
        "avg_ram": avg_ram,
    }


def main():
    # ✅ Update this to your actual S3 dataset path
    s3_path = "s3://authenta-streaming-data/original_dataset/dragon_train_000/"
    cache_path = os.path.expanduser("~/.ray/data")

    print(f"📂 Loading dataset from S3: {s3_path}")

    num_runs = 1
    print(f"\n🚀 Starting {num_runs} benchmark iteration(s)...")

    all_metrics = []
    for run_id in range(num_runs):
        metrics = benchmark_ray_direct_loading(s3_path, cache_path, run_id)
        all_metrics.append(metrics)

        print(f"\n📊 Run {run_id + 1} Results:")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Total images: {metrics['total_images']}")
        print(f"  Images per second: {metrics['images_per_sec']:.2f}")
        print(f"  Avg data load time: {metrics['avg_data_load_time']:.4f}s")

        ray.shutdown()
        time.sleep(2)

    # Aggregate stats
    metrics_keys = [
        "total_time",
        "total_images",
        "images_per_sec",
        "avg_data_load_time",
        "avg_cpu",
        "avg_ram",
    ]
    stats = {
        key: {"mean": np.mean([r[key] for r in all_metrics]),
              "std": np.std([r[key] for r in all_metrics])}
        for key in metrics_keys
    }

    os.makedirs("metrics", exist_ok=True)
    metrics_file = "metrics/ray-data_metrics_s3_images.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id"] + metrics_keys)
        for i, m in enumerate(all_metrics):
            writer.writerow([i + 1] + [m[k] for k in metrics_keys])
        writer.writerow([])
        writer.writerow(["statistic"] + metrics_keys)
        writer.writerow(["mean"] + [stats[k]["mean"] for k in metrics_keys])
        writer.writerow(["std"] + [stats[k]["std"] for k in metrics_keys])

    # Print summary
    print("\n" + "=" * 60)
    print(f"📈 BENCHMARK SUMMARY STATISTICS ({num_runs} run)")
    print("=" * 60)
    for key in ["total_time", "images_per_sec", "avg_data_load_time", "avg_cpu", "avg_ram"]:
        print(f"{key:>20}: {stats[key]['mean']:.2f} ± {stats[key]['std']:.2f}")

    print(f"\n💾 Results saved to: {metrics_file}")
    ray.shutdown()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
