import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    pass
plt.rcParams['figure.figsize'] = (16, 12)

# Load metrics from CSV files
metrics_dir = Path('/Volumes/Software/web_dataset 2/sample/metrics')

datasets = {
    'Ray Data': metrics_dir / 'ray-data_metrics_10000_images.csv',
    'Squirrel Data': metrics_dir / 'squirrel-data_metrics_10000_images.csv',
    'Streaming': metrics_dir / 'streaming_metrics_10000_images.csv',
    'TorchData': metrics_dir / 'torchdata_metrics_10000_images.csv',
    'WDS Data': metrics_dir / 'wds-data_metrics_10000_images.csv'
}

# Parse CSV files
all_stats = {}

for name, filepath in datasets.items():
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        continue
    
    with open(filepath, 'r') as f:
        # Detect delimiter (comma or tab)
        first_line = f.readline()
        f.seek(0)
        delimiter = '\t' if '\t' in first_line else ','
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
    
    # Find the mean row
    for row in rows:
        if row.get('run_id', '').strip() == 'mean':
            all_stats[name] = {
                'total_time': float(row['total_time']),
                'images_per_sec': float(row['images_per_sec']),
                'avg_data_load_time': float(row['avg_data_load_time']),
                'avg_cpu': float(row['avg_cpu']),
                'avg_ram': float(row['avg_ram'])
            }
            break

# Extract data for plotting
names = list(all_stats.keys())
total_times = [all_stats[name]['total_time'] for name in names]
imgs_per_sec = [all_stats[name]['images_per_sec'] for name in names]
avg_cpu = [all_stats[name]['avg_cpu'] for name in names]
avg_ram = [all_stats[name]['avg_ram'] for name in names]
data_load_times = [all_stats[name]['avg_data_load_time'] for name in names]
data_load_times_us = [t * 1e6 for t in data_load_times]
efficiency = [imgs_per_sec[i] / total_times[i] for i in range(len(names))]

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#95E1D3']

# 1. Total Time Comparison
ax1 = plt.subplot(2, 3, 1)
bars1 = ax1.bar(names, total_times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Total Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Total Processing Time', fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(total_times) * 1.15)
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{total_times[i]:.1f}s',
             ha='center', va='bottom', fontweight='bold')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# 2. Images Per Second Comparison (Higher is Better)
ax2 = plt.subplot(2, 3, 2)
imgs_per_sec = [all_stats[name]['images_per_sec'] for name in names]
bars2 = ax2.bar(names, imgs_per_sec, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Images/Second', fontsize=12, fontweight='bold')
ax2.set_title('Throughput (Higher is Better)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(imgs_per_sec) * 1.15)
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{imgs_per_sec[i]:.1f}',
             ha='center', va='bottom', fontweight='bold')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# 3. Average CPU Usage
ax3 = plt.subplot(2, 3, 3)
avg_cpu = [all_stats[name]['avg_cpu'] for name in names]
bars3 = ax3.bar(names, avg_cpu, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('CPU Usage (%)', fontsize=12, fontweight='bold')
ax3.set_title('Average CPU Usage', fontsize=13, fontweight='bold')
ax3.set_ylim(0, max(avg_cpu) * 1.15)
for i, bar in enumerate(bars3):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{avg_cpu[i]:.1f}%',
             ha='center', va='bottom', fontweight='bold')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)

# 4. Average RAM Usage
ax4 = plt.subplot(2, 3, 4)
avg_ram = [all_stats[name]['avg_ram'] for name in names]
bars4 = ax4.bar(names, avg_ram, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylabel('RAM Usage (GB)', fontsize=12, fontweight='bold')
ax4.set_title('Average Memory Usage', fontsize=13, fontweight='bold')
ax4.set_ylim(0, max(avg_ram) * 1.15)
for i, bar in enumerate(bars4):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{avg_ram[i]:.3f}GB',
             ha='center', va='bottom', fontweight='bold')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)

# 5. Data Load Time
ax5 = plt.subplot(2, 3, 5)
bars5 = ax5.bar(names, data_load_times_us, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_ylabel('Time (microseconds)', fontsize=12, fontweight='bold')
ax5.set_title('Average Data Load Time', fontsize=13, fontweight='bold')
ax5.set_ylim(0, max(data_load_times_us) * 1.15)
for i, bar in enumerate(bars5):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{data_load_times_us[i]:.2f}μs',
             ha='center', va='bottom', fontweight='bold', fontsize=9)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax5.grid(axis='y', alpha=0.3)

# 6. Efficiency Score
ax6 = plt.subplot(2, 3, 6)
bars6 = ax6.bar(names, efficiency, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('Efficiency Score', fontsize=12, fontweight='bold')
ax6.set_title('Throughput per Unit Time', fontsize=13, fontweight='bold')
ax6.set_ylim(0, max(efficiency) * 1.15)
for i, bar in enumerate(bars6):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{efficiency[i]:.2f}',
             ha='center', va='bottom', fontweight='bold')
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/Volumes/Software/web_dataset 2/sample/metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metrics_comparison.png")

# Create a detailed comparison table
fig2, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

# Prepare data for table
table_data = []
table_data.append(['Metric', 'Ray Data', 'Squirrel Data', 'Streaming', 'TorchData', 'WDS Data', 'Best'])

# Total Time
best_time = min(total_times)
row = ['Total Time (s)']
for name in names:
    val = all_stats[name]['total_time']
    marker = '✓' if val == best_time else ''
    row.append(f'{val:.2f} {marker}')
row.append('Lower')
table_data.append(row)

# Images Per Second
best_speed = max(imgs_per_sec)
row = ['Throughput (img/s)']
for name in names:
    val = all_stats[name]['images_per_sec']
    marker = '✓' if val == best_speed else ''
    row.append(f'{val:.1f} {marker}')
row.append('Higher')
table_data.append(row)

# CPU Usage
row = ['Avg CPU (%)']
for name in names:
    val = all_stats[name]['avg_cpu']
    row.append(f'{val:.1f}%')
row.append('Lower')
table_data.append(row)

# RAM Usage
best_ram = min(avg_ram)
row = ['Avg RAM (GB)']
for name in names:
    val = all_stats[name]['avg_ram']
    marker = '✓' if val == best_ram else ''
    row.append(f'{val:.3f} {marker}')
row.append('Lower')
table_data.append(row)

# Data Load Time
best_load = min(data_load_times)
row = ['Data Load Time (μs)']
for name in names:
    val = all_stats[name]['avg_data_load_time']
    marker = '✓' if val == best_load else ''
    row.append(f'{val*1e6:.2f} {marker}')
row.append('Lower')
table_data.append(row)

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.13, 0.15, 0.13, 0.13, 0.13, 0.10])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(7):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(6):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')
        table[(i, j)].set_text_props(weight='bold')

plt.title('Detailed Metrics Comparison Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig('/Volumes/Software/web_dataset 2/sample/metrics_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metrics_table.png")

# Print summary to console
print("\n" + "="*105)
print("METRICS COMPARISON SUMMARY (10,000 Images)")
print("="*105)
print(f"\n{'Metric':<25} {'Ray Data':<20} {'Squirrel Data':<20} {'Streaming':<20} {'TorchData':<20} {'WDS Data':<20}")
print("-"*130)
print(f"{'Total Time (s)':<25} {all_stats['Ray Data']['total_time']:<20.2f} {all_stats['Squirrel Data']['total_time']:<20.2f} {all_stats['Streaming']['total_time']:<20.2f} {all_stats['TorchData']['total_time']:<20.2f} {all_stats['WDS Data']['total_time']:<20.2f}")
print(f"{'Throughput (img/s)':<25} {all_stats['Ray Data']['images_per_sec']:<20.1f} {all_stats['Squirrel Data']['images_per_sec']:<20.1f} {all_stats['Streaming']['images_per_sec']:<20.1f} {all_stats['TorchData']['images_per_sec']:<20.1f} {all_stats['WDS Data']['images_per_sec']:<20.1f}")
print(f"{'Avg CPU (%)':<25} {all_stats['Ray Data']['avg_cpu']:<20.1f} {all_stats['Squirrel Data']['avg_cpu']:<20.1f} {all_stats['Streaming']['avg_cpu']:<20.1f} {all_stats['TorchData']['avg_cpu']:<20.1f} {all_stats['WDS Data']['avg_cpu']:<20.1f}")
print(f"{'Avg RAM (GB)':<25} {all_stats['Ray Data']['avg_ram']:<20.3f} {all_stats['Squirrel Data']['avg_ram']:<20.3f} {all_stats['Streaming']['avg_ram']:<20.3f} {all_stats['TorchData']['avg_ram']:<20.3f} {all_stats['WDS Data']['avg_ram']:<20.3f}")
print(f"{'Data Load Time (μs)':<25} {all_stats['Ray Data']['avg_data_load_time']*1e6:<20.2f} {all_stats['Squirrel Data']['avg_data_load_time']*1e6:<20.2f} {all_stats['Streaming']['avg_data_load_time']*1e6:<20.2f} {all_stats['TorchData']['avg_data_load_time']*1e6:<20.2f} {all_stats['WDS Data']['avg_data_load_time']*1e6:<20.2f}")
print("-"*130)

# Ranking
print("\n" + "="*105)
print("RANKINGS")
print("="*105)

rankings = {
    'Speed (Throughput)': sorted(zip(names, imgs_per_sec), key=lambda x: x[1], reverse=True),
    'Time (Total)': sorted(zip(names, total_times), key=lambda x: x[1]),
    'Memory (RAM)': sorted(zip(names, avg_ram), key=lambda x: x[1]),
    'CPU Usage': sorted(zip(names, avg_cpu), key=lambda x: x[1]),
    'Data Load Speed': sorted(zip(names, data_load_times), key=lambda x: x[1]),
}

for metric, ranking in rankings.items():
    print(f"\n{metric}:")
    for idx, (name, value) in enumerate(ranking, 1):
        print(f"  {idx}. {name}")

print("\n" + "="*105)
print("CONCLUSION")
print("="*105)

best_throughput = max(imgs_per_sec)
best_throughput_name = names[imgs_per_sec.index(best_throughput)]
best_time_val = min(total_times)
best_time_name = names[total_times.index(best_time_val)]
best_ram_val = min(avg_ram)
best_ram_name = names[avg_ram.index(best_ram_val)]

print(f"""
🏆 BEST PERFORMERS:

   ⚡ Speed (Throughput): {best_throughput_name}
      - {best_throughput:.1f} images/sec

   ⏱️  Total Time: {best_time_name}
      - {best_time_val:.2f} seconds

   💾 Memory Efficiency: {best_ram_name}
      - {best_ram_val:.3f}GB

📊 KEY INSIGHTS:
   - {best_throughput_name} achieves ~{best_throughput / min([v for i, v in enumerate(imgs_per_sec) if i != imgs_per_sec.index(best_throughput)]):.1f}x higher throughput than the slowest option
   - Squirrel Data offers excellent memory efficiency
   - TorchData provides balanced performance across metrics
   
⚙️  RECOMMENDATION:
   Choose based on your priorities:
   • Best Overall Speed: {best_throughput_name}
   • Best Memory Usage: {best_ram_name}
   • Best for Low-Latency: {best_time_name}
   • Balanced Choice: Consider Ray Data or TorchData
""")
print("="*105)

plt.show()
