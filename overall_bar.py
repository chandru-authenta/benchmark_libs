"""overall_bar.py

Simple professional horizontal bar chart showing overall scores (percentage) for each
data-loading solution. Outputs a PNG to the metrics folder and prints the winner.

Usage:
    python overall_bar.py

Outputs:
    overall_score_bar.png (in the same folder)
"""
import csv
from pathlib import Path
import math

# Try importing matplotlib; if missing, show a helpful message
try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as e:
    print("Matplotlib (and possibly numpy) is required to run this script.")
    print("Install with: pip install matplotlib numpy")
    raise

# Config
BASE_DIR = Path(__file__).resolve().parent
METRICS_DIR = BASE_DIR / 'metrics'
OUTPUT_PNG = BASE_DIR / 'overall_score_bar.png'

DATASETS = {
    'Ray Data': METRICS_DIR / 'ray-data_metrics_10000_images.csv',
    'Squirrel Data': METRICS_DIR / 'squirrel-data_metrics_10000_images.csv',
    'Streaming': METRICS_DIR / 'streaming_metrics_10000_images.csv',
    'TorchData': METRICS_DIR / 'torchdata_metrics_10000_images.csv',
    'WDS Data': METRICS_DIR / 'wds-data_metrics_10000_images.csv',
}

WEIGHTS = {
    'speed': 0.4,      # throughput (images/sec)
    'efficiency': 0.4, # total_time (lower better)
    'memory': 0.1,     # avg_ram (lower better)
    'cpu': 0.1,        # avg_cpu (lower better)
}

# Helper: read 'mean' row robustly with comma or tab delimiter
def read_mean_stats(path: Path):
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()
        f.seek(0)
        delim = '\t' if '\t' in first else ','
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            run_id = (row.get('run_id') or '').strip()
            if run_id.lower() == 'mean' or run_id.lower() == 'average':
                # ensure numeric conversion; return dict of floats where possible
                out = {}
                for k, v in row.items():
                    if v is None:
                        out[k] = None
                        continue
                    s = v.strip()
                    # try float conversion
                    try:
                        out[k] = float(s)
                    except Exception:
                        out[k] = s
                return out
    return None

# Load all stats
all_stats = {}
for name, path in DATASETS.items():
    stat = read_mean_stats(path)
    if stat is None:
        print(f"Warning: no mean row found for {name} in {path}")
        continue
    # Ensure the keys we need exist
    try:
        total_time = float(stat['total_time'])
        images_per_sec = float(stat['images_per_sec'])
        avg_ram = float(stat['avg_ram'])
        avg_cpu = float(stat['avg_cpu'])
    except Exception as e:
        print(f"Missing or invalid numeric columns for {name}: {e}")
        continue
    all_stats[name] = {
        'total_time': total_time,
        'images_per_sec': images_per_sec,
        'avg_ram': avg_ram,
        'avg_cpu': avg_cpu,
    }

if not all_stats:
    raise SystemExit('No valid stats found; please check CSV files in metrics folder.')

# Compute normalization baselines
names = list(all_stats.keys())
imgs = [all_stats[n]['images_per_sec'] for n in names]
times = [all_stats[n]['total_time'] for n in names]
rams = [all_stats[n]['avg_ram'] for n in names]
cpus = [all_stats[n]['avg_cpu'] for n in names]

max_throughput = max(imgs)
min_time = min(times)
min_ram = min(rams)
min_cpu = min(cpus)

# Score computation (0-100). For 'lower is better' metrics we invert (min/val*100)
scores = {}
for n in names:
    s = all_stats[n]
    speed_score = (s['images_per_sec'] / max_throughput) * 100 if max_throughput > 0 else 0
    eff_score = (min_time / s['total_time']) * 100 if s['total_time'] > 0 else 0
    mem_score = (min_ram / s['avg_ram']) * 100 if s['avg_ram'] > 0 else 0
    cpu_score = (min_cpu / s['avg_cpu']) * 100 if s['avg_cpu'] > 0 else 0
    overall = (
        speed_score * WEIGHTS['speed'] +
        eff_score * WEIGHTS['efficiency'] +
        mem_score * WEIGHTS['memory'] +
        cpu_score * WEIGHTS['cpu']
    )
    # Clamp and round
    overall = max(0.0, min(100.0, overall))
    scores[n] = overall

# Sort by score descending
sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
sorted_names = [s[0] for s in sorted_items]
sorted_scores = [s[1] for s in sorted_items]

# Produce single horizontal bar chart
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass  # Use default style
fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = plt.get_cmap('tab10').colors
# Map colors in order
colors = [bar_colors[i % len(bar_colors)] for i in range(len(sorted_names))]

y_pos = range(len(sorted_names))
ax.barh(y_pos, sorted_scores, color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names, fontsize=12)
ax.invert_yaxis()  # highest on top
ax.set_xlim(0, 100)
ax.set_xlabel('Overall Score (%)', fontsize=12)
ax.set_title('Overall Benchmark - Single Score Comparison', fontsize=14, fontweight='bold')

# Add percentage labels on bars
for i, v in enumerate(sorted_scores):
    ax.text(v + 1.5, i, f'{v:.1f}%', va='center', fontweight='bold')

# Add a subtle grid and tidy layout
ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()

# Save output
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"Saved overall bar chart: {OUTPUT_PNG}")

# Print winner summary
winner = sorted_names[0]
print('\nOVERALL WINNER:')
print(f"  {winner} — {scores[winner]:.1f}%")
print('\nShort metrics:')
w = all_stats[winner]
print(f"  Throughput: {w['images_per_sec']:.1f} img/s | Total time: {w['total_time']:.2f}s | RAM: {w['avg_ram']:.3f}GB | CPU: {w['avg_cpu']:.1f}%")

# If run interactively, show plot
if hasattr(plt, 'show'):
    try:
        plt.show()
    except Exception:
        # headless environment
        pass

if __name__ == '__main__':
    pass
