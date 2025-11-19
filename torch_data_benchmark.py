import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import time
import psutil
import csv

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.paths = []
        
        # Walk through directory recursively
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    self.paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def state_dict(self):
        # if you have custom RNGs etc.
        return {}

    def load_state_dict(self, state_dict):
        # restore custom state if needed
        pass

# Then use your custom sampler (like you had)
class MySampler(Sampler[int]):
    def __init__(self, high, seed=None, limit=None):
        self.high = high
        self.seed = seed
        self.limit = limit if limit is not None else high
        self.current = 0
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __iter__(self):
        self.current = 0
        perm = torch.randperm(self.high, generator=self.rng)
        return iter(perm[:self.limit].tolist())

    def __len__(self):
        return self.limit

    def state_dict(self):
        return {"current": self.current, "rng_state": self.rng.get_state()}

    def load_state_dict(self, state_dict):
        self.current = state_dict["current"]
        self.rng.set_state(state_dict["rng_state"])


def torchdata_direct_loading(dataset, sampler, run_id):

    print(f"\n🔄 Running benchmark iteration {run_id + 1}...")
    process = psutil.Process(os.getpid())

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=32,
        drop_last=False,
        num_workers=4
    )

    print("🔎 Measuring dataset loading performance...")
    all_data_load_times = []
    all_cpu, all_ram, all_disk = [], [], []
    total_images = 0

    def get_metrics():
        cpu = psutil.cpu_percent(interval=None)
        ram = process.memory_info().rss / (1024**3)
        try:
            disk = psutil.disk_usage(dataset).used / (1024**3)
        except Exception:
            disk = psutil.disk_usage("/").used / (1024**3)
        return cpu, ram, disk

    start_time = time.time()
    batch_idx = 0
    batch_time=time.time()
    for batch in loader:
        elapsed=time.time()-batch_time


        all_data_load_times.append(time.time() - elapsed)
        batch_idx+=1
        total_images += len(batch)
        batch_time=time.time()
        print(f"process batch of {len(batch)} images")

        # Collect metrics
        cpu, ram, disk = get_metrics()
        all_disk.append(disk)
        all_cpu.append(cpu)
        all_ram.append(ram)

    total_time = time.time() - start_time

    # Return metrics for this run
    return {
        "total_time": total_time,
        "total_images": total_images,
        "images_per_sec": total_images / total_time,
        "avg_data_load_time": np.mean(all_data_load_times),
        "avg_cpu": np.mean(all_cpu),
        "avg_ram": np.mean(all_ram),
    }



def main():

    local = '/home/ubuntu/s3mount/original_dataset'
    print(f"Loading dataset from: {local}")

    # Number of benchmark runs
    num_runs = 1
    print(f"\n🚀 Starting {num_runs} benchmark iterations...")

    dataset = ImageFolderDataset(local,
        transform=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ]))
    sampler = MySampler(high=len(dataset), seed=42)

    # Collect metrics from all runs
    all_metrics = []

    for run_id in range(num_runs):
        metrics = torchdata_direct_loading(dataset, sampler, run_id)
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
    ]

    stats = {}
    for key in metrics_keys:
        values = [run[key] for run in all_metrics]
        stats[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }

    # Save detailed metrics to CSV
    metrics_file = "ec2_metrics/dataloader_metrics.csv"
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



