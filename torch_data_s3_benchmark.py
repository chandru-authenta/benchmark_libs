import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision import transforms
from PIL import Image
import s3fs
import io
import os
import numpy as np
import psutil
import time
import csv
from threading import Lock

# Global S3 connection pool to avoid reconnecting
_s3_pool = {}
_s3_lock = Lock()

def get_s3_connection(anon=False):
    """Get or create a persistent S3 connection (thread-safe)."""
    worker_id = os.getpid()
    
    with _s3_lock:
        if worker_id not in _s3_pool:
            try:
                # Create connection with optimized settings
                _s3_pool[worker_id] = s3fs.S3FileSystem(
                    anon=anon,
                    use_ssl=True,
                    requester_pays=False,
                    skip_instance_cache=False,  # Enable caching
                    config_kwargs={
                        'retries': {'max_attempts': 3, 'mode': 'adaptive'},
                        'read_timeout': 60,
                        'connect_timeout': 10,
                    }
                )
            except Exception as e:
                print(f"⚠️ Failed to create S3 connection for worker {worker_id}: {repr(e)}")
                # Fallback to basic connection
                _s3_pool[worker_id] = s3fs.S3FileSystem(anon=anon)
        return _s3_pool[worker_id]


class S3ImageStreamDataset(IterableDataset):
    def __init__(self, s3_prefix, transform=None):
        super().__init__()
        self.s3_prefix = s3_prefix.rstrip("/")
        self.transform = transform
        self._files_cache = None
    
    @property
    def files(self):
        """Lazy-load and cache file list."""
        if self._files_cache is None:
            # Get persistent connection for file listing
            fs = get_s3_connection(anon=False)
            
            # Find all files under prefix (recursive) with retries
            max_retries = 3
            all_files = []
            
            for attempt in range(max_retries):
                try:
                    all_files = fs.find(self.s3_prefix)
                    break
                except Exception as e:
                    if self.s3_prefix.startswith("s3://"):
                        try:
                            without_scheme = self.s3_prefix[5:]
                            all_files = fs.find(without_scheme)
                            break
                        except Exception as e2:
                            if attempt == max_retries - 1:
                                raise RuntimeError(f"Failed to list S3 files after {max_retries} attempts: {repr(e2)}")
                            print(f"⚠️ Attempt {attempt + 1} failed, retrying: {repr(e2)}")
                            time.sleep(1)
                    else:
                        if attempt == max_retries - 1:
                            raise RuntimeError(f"Failed to list S3 files after {max_retries} attempts: {repr(e)}")
                        print(f"⚠️ Attempt {attempt + 1} failed, retrying: {repr(e)}")
                        time.sleep(1)
            
            # Filter by extensions (case-insensitive)
            exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
            self._files_cache = sorted([f for f in all_files if any(f.lower().endswith(ext) for ext in exts)])
            print(f"✅ Found {len(self._files_cache)} S3 images under: {self.s3_prefix}")
        
        return self._files_cache

    def __iter__(self):
        # Get persistent connection for this worker
        fs = get_s3_connection(anon=False)
        
        worker_info = get_worker_info()
        if worker_info is None:
            files = self.files
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
           
            chunks = np.array_split(np.array(self.files), num_workers)
            files = list(chunks[worker_id]) if len(chunks) > worker_id else []

        for key in files:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Reuse persistent connection
                    with fs.open(key, "rb") as f:
                        img_bytes = f.read()
                    
                    if not img_bytes:
                        print(f"⚠️ Skipping empty file: {key}")
                        break
                    
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    yield img
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"⚠️ Failed to load {key} after {max_retries} attempts: {repr(e)}")
                    else:
                        print(f"⚠️ Retry {retry_count}/{max_retries} for {key}: {repr(e)}")
                        time.sleep(0.1 * retry_count)  # Exponential backoff
                    continue

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def worker_init_fn(worker_id):
    """Initialize S3 connection for each worker at startup."""
    try:
        fs = get_s3_connection(anon=False)
        print(f"✅ Worker {worker_id} initialized S3 connection")
    except Exception as e:
        print(f"⚠️ Worker {worker_id} failed to initialize: {repr(e)}")


def run_benchmark(dataset, run_id, num_workers=4, batch_size=32):
    """Run benchmark with improved error handling and timing."""
    print(f"\n🔄 Running benchmark iteration {run_id + 1}...")
    process = psutil.Process(os.getpid())

    loader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=4,  # Increased prefetch
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )

    print("🔎 Measuring dataset loading performance...")
    all_data_load_times = []
    all_cpu, all_ram = [], []
    total_images = 0

    def get_metrics():
        cpu = psutil.cpu_percent(interval=None)
        ram = process.memory_info().rss / (1024**3)
        try:
            disk = psutil.disk_usage("/").used / (1024**3)
        except Exception:
            disk = float("nan")
        return cpu, ram, disk

    start_time = time.time()
    batch_idx = 0
    batch_start = time.time()
    
    for batch in loader:
        elapsed = time.time() - batch_start
        all_data_load_times.append(elapsed)

        batch_idx += 1
        try:
            batch_len = len(batch)
        except Exception:
            batch_len = 1
        total_images += batch_len
        print(f"🖼️ processed batch {batch_idx} with {batch_len} images (load took {elapsed:.4f}s)")

        cpu, ram, disk = get_metrics()
        all_cpu.append(cpu)
        all_ram.append(ram)
        batch_start = time.time()
    
    total_time = time.time() - start_time

    if len(all_data_load_times) == 0:
        avg_load = float("nan")
    else:
        avg_load = float(np.mean(all_data_load_times))

    return {
        "total_time": total_time,
        "total_images": total_images,
        "images_per_sec": (total_images / total_time) if total_time > 0 and total_images > 0 else 0.0,
        "avg_data_load_time": avg_load,
        "avg_cpu": float(np.mean(all_cpu)) if all_cpu else float("nan"),
        "avg_ram": float(np.mean(all_ram)) if all_ram else float("nan"),
    }


def main():
    s3_uri = "s3://authenta-streaming-data/original_dataset/dragon_train_000"
    print(f"Loading dataset from: {s3_uri}")

    num_runs = 1
    print(f"\n🚀 Starting {num_runs} benchmark runs...")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = S3ImageStreamDataset(s3_uri, transform=transform)

    all_metrics = []
    for run_id in range(num_runs):
        metrics = run_benchmark(dataset, run_id, num_workers=4, batch_size=32)
        all_metrics.append(metrics)

        print(f"\n📊 Run {run_id + 1} Results:")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Total images: {metrics['total_images']}")
        print(f"  Images per second: {metrics['images_per_sec']:.2f}")
        print(f"  Avg data load time: {metrics['avg_data_load_time']:.4f}s")

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

    metrics_file = "metrics/torchdata_metrics_s3_images.csv"
    os.makedirs("metrics", exist_ok=True)
    
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id"] + metrics_keys)
        
        for i, metrics in enumerate(all_metrics):
            writer.writerow([i + 1] + [metrics[key] for key in metrics_keys])

        writer.writerow([])
        writer.writerow(["statistic"] + metrics_keys)
        writer.writerow(["mean"] + [stats[key]["mean"] for key in metrics_keys])
        writer.writerow(["std"] + [stats[key]["std"] for key in metrics_keys])

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