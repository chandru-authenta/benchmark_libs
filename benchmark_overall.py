import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    pass

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

# Calculate scores (0-100 scale, higher is better)
scores = {}
for name in all_stats:
    scores[name] = {
        'speed': 0,
        'efficiency': 0,
        'memory': 0,
        'cpu': 0,
        'load_time': 0,
        'overall': 0
    }

# Get min/max for normalization
names = list(all_stats.keys())
total_times = [all_stats[name]['total_time'] for name in names]
imgs_per_sec = [all_stats[name]['images_per_sec'] for name in names]
avg_cpu = [all_stats[name]['avg_cpu'] for name in names]
avg_ram = [all_stats[name]['avg_ram'] for name in names]
data_load_times = [all_stats[name]['avg_data_load_time'] for name in names]

max_throughput = max(imgs_per_sec)
min_time = min(total_times)
min_cpu = min(avg_cpu)
min_ram = min(avg_ram)
min_load_time = min(data_load_times)

# Calculate individual scores (higher metrics = higher score, lower metrics = need inversion)
for name in names:
    stats = all_stats[name]
    
    # Speed score (throughput) - Higher is better
    scores[name]['speed'] = (stats['images_per_sec'] / max_throughput) * 100
    
    # Efficiency score (time) - Lower is better, so invert
    scores[name]['efficiency'] = (min_time / stats['total_time']) * 100
    
    # Memory score - Lower is better, so invert
    scores[name]['memory'] = (min_ram / stats['avg_ram']) * 100
    
    # CPU score - Lower is better, so invert
    scores[name]['cpu'] = (min_cpu / stats['avg_cpu']) * 100
    
    # Load time score - Lower is better, so invert
    scores[name]['load_time'] = (min_load_time / stats['avg_data_load_time']) * 100
    
    # Overall score - weighted average (priority: speed=35%, efficiency=35%, memory=15%, cpu=10%, load_time=5%)
    scores[name]['overall'] = (
        scores[name]['speed'] * 0.35 +
        scores[name]['efficiency'] * 0.35 +
        scores[name]['memory'] * 0.15 +
        scores[name]['cpu'] * 0.10 +
        scores[name]['load_time'] * 0.05
    )

# Sort by overall score
sorted_solutions = sorted(scores.items(), key=lambda x: x[1]['overall'], reverse=True)

# Create comprehensive benchmark figure
fig = plt.figure(figsize=(20, 14))

# 1. OVERALL WINNER - Big visual impact
ax1 = plt.subplot(3, 3, 1)
ax1.axis('off')

winner_name = sorted_solutions[0][0]
winner_score = sorted_solutions[0][1]['overall']

# Create a prominent winner badge
circle = plt.Circle((0.5, 0.7), 0.35, color='#FFD700', alpha=0.3, zorder=1)
ax1.add_patch(circle)
ax1.text(0.5, 0.75, '🏆', fontsize=80, ha='center', va='center', zorder=2)
ax1.text(0.5, 0.35, winner_name, fontsize=28, ha='center', va='center', fontweight='bold', zorder=2)
ax1.text(0.5, 0.1, f'Score: {winner_score:.1f}/100', fontsize=22, ha='center', va='center', 
         fontweight='bold', color='#2C3E50', zorder=2)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# 2. Overall Scores Leaderboard
ax2 = plt.subplot(3, 3, 2)
rank_names = [s[0] for s in sorted_solutions]
rank_scores = [s[1]['overall'] for s in sorted_solutions]
colors_rank = ['#FFD700', '#C0C0C0', '#CD7F32', '#4ECDC4', '#95E1D3']
bars = ax2.barh(rank_names, rank_scores, color=colors_rank, edgecolor='black', linewidth=2)
ax2.set_xlabel('Overall Score', fontsize=12, fontweight='bold')
ax2.set_title('🏅 OVERALL RANKINGS', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 110)
for i, (bar, score) in enumerate(zip(bars, rank_scores)):
    ax2.text(score + 2, bar.get_y() + bar.get_height()/2, f'{score:.1f}', 
             va='center', fontweight='bold', fontsize=11)
ax2.grid(axis='x', alpha=0.3)

# 3. Speed Score
ax3 = plt.subplot(3, 3, 3)
speed_data = sorted([(name, scores[name]['speed']) for name in names], 
                    key=lambda x: x[1], reverse=True)
speed_names = [s[0] for s in speed_data]
speed_scores = [s[1] for s in speed_data]
ax3.bar(range(len(speed_names)), speed_scores, color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_xticks(range(len(speed_names)))
ax3.set_xticklabels(speed_names, rotation=45, ha='right')
ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('⚡ Speed Score (35% weight)', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 110)
for i, score in enumerate(speed_scores):
    ax3.text(i, score + 2, f'{score:.0f}', ha='center', fontweight='bold', fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# 4. Efficiency Score
ax4 = plt.subplot(3, 3, 4)
eff_data = sorted([(name, scores[name]['efficiency']) for name in names], 
                  key=lambda x: x[1], reverse=True)
eff_names = [s[0] for s in eff_data]
eff_scores = [s[1] for s in eff_data]
ax4.bar(range(len(eff_names)), eff_scores, color='#45B7D1', alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_xticks(range(len(eff_names)))
ax4.set_xticklabels(eff_names, rotation=45, ha='right')
ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
ax4.set_title('⏱️  Efficiency Score (35% weight)', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 110)
for i, score in enumerate(eff_scores):
    ax4.text(i, score + 2, f'{score:.0f}', ha='center', fontweight='bold', fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# 5. Memory Score
ax5 = plt.subplot(3, 3, 5)
mem_data = sorted([(name, scores[name]['memory']) for name in names], 
                  key=lambda x: x[1], reverse=True)
mem_names = [s[0] for s in mem_data]
mem_scores = [s[1] for s in mem_data]
ax5.bar(range(len(mem_names)), mem_scores, color='#FFA07A', alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_xticks(range(len(mem_names)))
ax5.set_xticklabels(mem_names, rotation=45, ha='right')
ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
ax5.set_title('💾 Memory Score (15% weight)', fontsize=12, fontweight='bold')
ax5.set_ylim(0, 110)
for i, score in enumerate(mem_scores):
    ax5.text(i, score + 2, f'{score:.0f}', ha='center', fontweight='bold', fontsize=10)
ax5.grid(axis='y', alpha=0.3)

# 6. CPU Score
ax6 = plt.subplot(3, 3, 6)
cpu_data = sorted([(name, scores[name]['cpu']) for name in names], 
                  key=lambda x: x[1], reverse=True)
cpu_names = [s[0] for s in cpu_data]
cpu_scores = [s[1] for s in cpu_data]
ax6.bar(range(len(cpu_names)), cpu_scores, color='#95E1D3', alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_xticks(range(len(cpu_names)))
ax6.set_xticklabels(cpu_names, rotation=45, ha='right')
ax6.set_ylabel('Score', fontsize=11, fontweight='bold')
ax6.set_title('🖥️  CPU Score (10% weight)', fontsize=12, fontweight='bold')
ax6.set_ylim(0, 110)
for i, score in enumerate(cpu_scores):
    ax6.text(i, score + 2, f'{score:.0f}', ha='center', fontweight='bold', fontsize=10)
ax6.grid(axis='y', alpha=0.3)

# 7. Load Time Score
ax7 = plt.subplot(3, 3, 7)
load_data = sorted([(name, scores[name]['load_time']) for name in names], 
                   key=lambda x: x[1], reverse=True)
load_names = [s[0] for s in load_data]
load_scores = [s[1] for s in load_data]
ax7.bar(range(len(load_names)), load_scores, color='#A8E6CF', alpha=0.7, edgecolor='black', linewidth=2)
ax7.set_xticks(range(len(load_names)))
ax7.set_xticklabels(load_names, rotation=45, ha='right')
ax7.set_ylabel('Score', fontsize=11, fontweight='bold')
ax7.set_title('⚙️  Load Time Score (5% weight)', fontsize=12, fontweight='bold')
ax7.set_ylim(0, 110)
for i, score in enumerate(load_scores):
    ax7.text(i, score + 2, f'{score:.0f}', ha='center', fontweight='bold', fontsize=10)
ax7.grid(axis='y', alpha=0.3)

# 8. Radar-like comparison (Score breakdown)
ax8 = plt.subplot(3, 3, 8)
metrics_cat = ['Speed', 'Efficiency', 'Memory', 'CPU', 'Load Time']
top_3 = [s[0] for s in sorted_solutions[:3]]
x_pos = np.arange(len(metrics_cat))
width = 0.25

for i, name in enumerate(top_3):
    values = [scores[name]['speed'], scores[name]['efficiency'], scores[name]['memory'], 
              scores[name]['cpu'], scores[name]['load_time']]
    ax8.bar(x_pos + i*width, values, width, label=name, alpha=0.8, edgecolor='black', linewidth=1.5)

ax8.set_ylabel('Score', fontsize=11, fontweight='bold')
ax8.set_title('📊 Top 3 Solutions Comparison', fontsize=12, fontweight='bold')
ax8.set_xticks(x_pos + width)
ax8.set_xticklabels(metrics_cat, fontsize=10)
ax8.legend(loc='upper right', fontsize=10)
ax8.set_ylim(0, 110)
ax8.grid(axis='y', alpha=0.3)

# 9. Summary Stats Table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_text = "🎯 KEY FINDINGS\n" + "="*45 + "\n\n"
for i, (name, score_dict) in enumerate(sorted_solutions):
    rank_emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i]
    summary_text += f"{rank_emoji} {name:<20} {score_dict['overall']:>6.1f}/100\n"

summary_text += "\n" + "="*45 + "\n"
summary_text += f"\n✅ BEST FOR:\n"
summary_text += f"  • Throughput: {max(imgs_per_sec) == scores[sorted_solutions[0][0]]['speed'] * max_throughput / 100 and sorted_solutions[0][0] or [n for n in names if imgs_per_sec[names.index(n)] == max(imgs_per_sec)][0]}\n"
summary_text += f"  • Memory: {[n for n in names if avg_ram[names.index(n)] == min(avg_ram)][0]}\n"
summary_text += f"  • Speed: {[n for n in names if total_times[names.index(n)] == min(total_times)][0]}\n"

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace', 
         bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8, edgecolor='black', linewidth=2))

plt.suptitle('📊 DATA LOADING SOLUTIONS - COMPREHENSIVE BENCHMARK REPORT 📊\n10,000 Images Dataset',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('/Volumes/Software/web_dataset 2/sample/benchmark_overall.png', dpi=300, bbox_inches='tight')
print("✓ Saved: benchmark_overall.png")

# Print comprehensive report
print("\n" + "="*100)
print(" "*25 + "📊 COMPREHENSIVE BENCHMARK REPORT 📊")
print(" "*20 + "Data Loading Solutions - 10,000 Images Dataset")
print("="*100)

print("\n🏆 OVERALL RANKINGS (Weighted Score):")
print("-"*100)
for i, (name, score_dict) in enumerate(sorted_solutions, 1):
    rank_emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i-1]
    print(f"{rank_emoji} #{i}. {name:<20} OVERALL SCORE: {score_dict['overall']:>6.1f}/100")
    print(f"    ├─ Speed (35%):      {score_dict['speed']:>6.1f}/100")
    print(f"    ├─ Efficiency (35%): {score_dict['efficiency']:>6.1f}/100")
    print(f"    ├─ Memory (15%):     {score_dict['memory']:>6.1f}/100")
    print(f"    ├─ CPU (10%):        {score_dict['cpu']:>6.1f}/100")
    print(f"    └─ Load Time (5%):   {score_dict['load_time']:>6.1f}/100")
print("-"*100)

print("\n📈 DETAILED METRICS COMPARISON:")
print("-"*100)
print(f"{'Solution':<20} {'Total Time':<15} {'Throughput':<15} {'Avg RAM':<15} {'Avg CPU':<15} {'Load Time':<15}")
print(f"{'':20} {'(seconds)':<15} {'(img/sec)':<15} {'(GB)':<15} {'(%)':<15} {'(μs)':<15}")
print("-"*100)
for name in [s[0] for s in sorted_solutions]:
    stats = all_stats[name]
    print(f"{name:<20} {stats['total_time']:<15.2f} {stats['images_per_sec']:<15.1f} {stats['avg_ram']:<15.3f} {stats['avg_cpu']:<15.1f} {stats['avg_data_load_time']*1e6:<15.2f}")
print("-"*100)

print("\n💡 RECOMMENDATIONS BY USE CASE:")
print("-"*100)
print("\n🚀 Best for PRODUCTION (High Throughput):")
best_speed = sorted([(name, scores[name]['speed']) for name in names], key=lambda x: x[1], reverse=True)
print(f"   → {best_speed[0][0]}")
print(f"     Reason: Highest throughput, best for maximum data processing speed")

print("\n💰 Best for COST-EFFECTIVE (Memory Efficiency):")
best_memory = sorted([(name, scores[name]['memory']) for name in names], key=lambda x: x[1], reverse=True)
print(f"   → {best_memory[0][0]}")
print(f"     Reason: Lowest memory footprint, best for resource-constrained environments")

print("\n⚖️  Best BALANCED Solution:")
print(f"   → {sorted_solutions[0][0]}")
print(f"     Reason: Best overall performance across all metrics ({sorted_solutions[0][1]['overall']:.1f}/100)")

print("\n🎯 Best for EDGE COMPUTING (Low CPU):")
best_cpu = sorted([(name, scores[name]['cpu']) for name in names], key=lambda x: x[1], reverse=True)
print(f"   → {best_cpu[0][0]}")
print(f"     Reason: Lowest CPU usage, best for lightweight deployments")

print("\n" + "="*100)
print("\n✅ SCORING METHODOLOGY:")
print("   - Speed Score (35%): Based on throughput (images/sec)")
print("   - Efficiency Score (35%): Based on total processing time")
print("   - Memory Score (15%): Based on RAM usage (lower is better)")
print("   - CPU Score (10%): Based on CPU usage (lower is better)")
print("   - Load Time Score (5%): Based on data load latency")
print("\n   Overall Score = Weighted average of all metrics")
print("="*100)

plt.show()
