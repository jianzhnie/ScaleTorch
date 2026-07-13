import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Use a Chinese-capable font on macOS
plt.rcParams['font.family'] = ['Heiti SC', 'Arial Unicode MS', 'sans-serif']

# Figure 17: 2D cluster topology
fig, ax = plt.subplots(figsize=(12, 8))

n_hosts = 8
gpus_per_host = 8
host_spacing = 1.2
gpu_spacing = 0.9

# Colors
gpu_color = '#bbdefb'
gpu_edge = '#1565c0'
nvlink_color = '#ef6c00'
dp_color = '#2e7d32'

# Draw hosts and GPUs
for h in range(n_hosts):
    host_y = (n_hosts - 1 - h) * host_spacing
    # Host label
    ax.text(-0.8, host_y,
            f'主机 {h}', fontsize=11, fontweight='bold', ha='right', va='center')

    # Draw NVLink connection line across GPUs
    nvlink_y = host_y - 0.2
    ax.plot([0, (gpus_per_host - 1) * gpu_spacing], [nvlink_y, nvlink_y],
            color=nvlink_color, linewidth=3, alpha=0.6)
    ax.text((gpus_per_host - 1) * gpu_spacing / 2, nvlink_y - 0.35,
            'TP (NVLink)', fontsize=9, ha='center', color=nvlink_color, fontweight='bold')

    for g in range(gpus_per_host):
        x = g * gpu_spacing
        y = host_y
        rect = patches.FancyBboxPatch((x - 0.3, y - 0.25), 0.6, 0.5,
                                      boxstyle="round,pad=0.02",
                                      facecolor=gpu_color, edgecolor=gpu_edge, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, f'GPU\n{g}', fontsize=7, ha='center', va='center')

# Draw DP connections (vertical lines connecting same GPU across hosts)
for g in range(gpus_per_host):
    x = g * gpu_spacing
    y_top = (n_hosts - 1) * host_spacing + 0.25
    y_bottom = -0.25
    ax.plot([x, x], [y_bottom, y_top], color=dp_color, linewidth=2, linestyle='--', alpha=0.5)

# Add DP label on the right
ax.text((gpus_per_host - 1) * gpu_spacing + 0.8, (n_hosts - 1) * host_spacing / 2,
        'FSDP2 / DP\n(跨主机网络)', fontsize=10, ha='left', va='center',
        color=dp_color, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor=dp_color, alpha=0.8))

# Title
ax.set_title('64 GPU 集群：8 主机 × 8 GPU\n主机内 NVLink 跑 TP，主机间网络跑 FSDP2',
             fontsize=14, fontweight='bold', pad=20)

ax.set_xlim(-1.5, (gpus_per_host - 1) * gpu_spacing + 2)
ax.set_ylim(-0.8, (n_hosts - 1) * host_spacing + 0.8)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_cluster_topology.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_cluster_topology.svg')

# Figure 21: 2D DeviceMesh
fig, ax = plt.subplots(figsize=(10, 10))

mesh_size = 8
cell_size = 1.0

# Colors
tp_color = '#fff3e0'
tp_edge = '#ef6c00'
dp_color = '#e8f5e9'
dp_edge = '#2e7d32'
grid_color = '#bdbdbd'

# Draw grid cells
for dp in range(mesh_size):
    for tp in range(mesh_size):
        x = tp * cell_size
        y = (mesh_size - 1 - dp) * cell_size
        rect = patches.Rectangle((x, y), cell_size, cell_size,
                                  facecolor='white', edgecolor=grid_color, linewidth=1)
        ax.add_patch(rect)
        ax.text(x + cell_size / 2, y + cell_size / 2,
                f'({dp},{tp})', fontsize=8, ha='center', va='center', color='#424242')

# Highlight tp_mesh columns
for tp in range(mesh_size):
    x = tp * cell_size
    y = 0
    rect = patches.Rectangle((x, y), cell_size, mesh_size * cell_size,
                              facecolor=tp_color, edgecolor=tp_edge, linewidth=2, alpha=0.3)
    ax.add_patch(rect)

# Highlight dp_mesh rows
for dp in range(mesh_size):
    x = 0
    y = (mesh_size - 1 - dp) * cell_size
    rect = patches.Rectangle((x, y), mesh_size * cell_size, cell_size,
                              facecolor=dp_color, edgecolor=dp_edge, linewidth=2, alpha=0.2)
    ax.add_patch(rect)

# Add axis labels
for tp in range(mesh_size):
    ax.text(tp * cell_size + cell_size / 2, mesh_size * cell_size + 0.15,
            f'tp={tp}', fontsize=9, ha='center', va='bottom', color='#1565c0', fontweight='bold')

for dp in range(mesh_size):
    ax.text(-0.15, (mesh_size - 1 - dp) * cell_size + cell_size / 2,
            f'dp={dp}', fontsize=9, ha='right', va='center', color='#1565c0', fontweight='bold')

# Legend
legend_x = mesh_size * cell_size + 0.5
legend_y = mesh_size * cell_size - 1.5
ax.add_patch(patches.Rectangle((legend_x, legend_y), 0.4, 0.4,
                                facecolor=tp_color, edgecolor=tp_edge, linewidth=2, alpha=0.5))
ax.text(legend_x + 0.5, legend_y + 0.2, 'tp_mesh: 纵向列 (主机内 TP)',
        fontsize=10, va='center')

ax.add_patch(patches.Rectangle((legend_x, legend_y - 0.8), 0.4, 0.4,
                                facecolor=dp_color, edgecolor=dp_edge, linewidth=2, alpha=0.5))
ax.text(legend_x + 0.5, legend_y - 0.6, 'dp_mesh: 横向行 (跨主机 FSDP2)',
        fontsize=10, va='center')

ax.set_title("mesh_2d = init_device_mesh('cuda', (8, 8))\n2-D DeviceMesh: [dp_size=8, tp_size=8]",
             fontsize=14, fontweight='bold', pad=20)

ax.set_xlim(-0.8, mesh_size * cell_size + 3.5)
ax.set_ylim(-0.8, mesh_size * cell_size + 0.8)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_device_mesh_2d.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_device_mesh_2d.svg')

# Figure 5: RowwiseParallel Embedding (improved styling)
fig, ax = plt.subplots(figsize=(13, 7))

n_rows, n_cols, n_gpus = 16, 8, 4
rows_per_gpu = n_rows // n_gpus
cell_w, cell_h = 0.55, 0.35
matrix_x, matrix_y = 2.8, 2.0
matrix_w = n_cols * cell_w
matrix_h = n_rows * cell_h
gpu_colors = ['#FFF9C4', '#FFF59D', '#FFF176', '#FFEE58']
edge_color = '#F9A825'

# Soft shadow behind matrix
shadow = patches.FancyBboxPatch(
    (matrix_x + 0.08, matrix_y - 0.08), matrix_w, matrix_h,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#E0E0E0', edgecolor='none', alpha=0.3)
ax.add_patch(shadow)

# Matrix cells
for gpu in range(n_gpus):
    for r in range(rows_per_gpu):
        row = gpu * rows_per_gpu + r
        for c in range(n_cols):
            x = matrix_x + c * cell_w
            y = matrix_y + (n_rows - 1 - row) * cell_h
            rect = patches.Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=gpu_colors[gpu], edgecolor=edge_color, linewidth=0.5)
            ax.add_patch(rect)

# Matrix border
border = patches.FancyBboxPatch(
    (matrix_x, matrix_y), matrix_w, matrix_h,
    boxstyle='round,pad=0.02,rounding_size=0.08',
    facecolor='none', edgecolor=edge_color, linewidth=2)
ax.add_patch(border)

# Matrix title
ax.text(matrix_x + matrix_w / 2, matrix_y + matrix_h + 0.45,
        'embedding_weight: [vocab_size, hidden_size]',
        fontsize=12, ha='center', va='bottom', fontweight='bold', color='#424242')

# Axis labels
ax.text(matrix_x - 0.55, matrix_y + matrix_h / 2,
        '词表维度\n(按行切分)', fontsize=10, ha='center', va='center',
        rotation=90, color='#424242')
ax.text(matrix_x + matrix_w / 2, matrix_y - 0.55,
        'hidden_size', fontsize=10, ha='center', va='top', color='#424242')

# GPU labels on the right (color chips + text)
for gpu in range(n_gpus):
    y_mid = matrix_y + matrix_h - (gpu * rows_per_gpu + rows_per_gpu / 2) * cell_h
    label_x = matrix_x + matrix_w + 0.6
    chip = patches.FancyBboxPatch(
        (label_x, y_mid - 0.3), 0.5, 0.6,
        boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=gpu_colors[gpu], edgecolor=edge_color, linewidth=1.5)
    ax.add_patch(chip)
    ax.text(label_x + 0.7, y_mid,
            f'GPU {gpu}    词表行 [{gpu * n_rows // n_gpus}:{(gpu + 1) * n_rows // n_gpus})',
            fontsize=10, va='center', ha='left', color='#424242', fontweight='bold')
    # subtle leader line
    ax.plot([matrix_x + matrix_w, label_x], [y_mid, y_mid],
            color=edge_color, linewidth=0.8, alpha=0.6)

# Input box
center_y = matrix_y + matrix_h / 2
input_rect = patches.FancyBboxPatch(
    (0.2, center_y - 0.6), 2.0, 1.2,
    boxstyle='round,pad=0.05,rounding_size=0.12',
    facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
ax.add_patch(input_rect)
ax.text(1.2, center_y, 'input_ids\n[batch, seq]\nReplicate',
        fontsize=10, ha='center', va='center', color='#212121', fontweight='bold')

# Output box
output_rect = patches.FancyBboxPatch(
    (matrix_x + matrix_w + 4.0, center_y - 0.6), 2.2, 1.2,
    boxstyle='round,pad=0.05,rounding_size=0.12',
    facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(output_rect)
ax.text(matrix_x + matrix_w + 5.1, center_y,
        'output\n[batch, seq, hidden]\nReplicate',
        fontsize=10, ha='center', va='center', color='#212121', fontweight='bold')

# Arrows
ax.annotate('', xy=(matrix_x - 0.05, center_y), xytext=(2.25, center_y),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
ax.annotate('', xy=(matrix_x + matrix_w + 3.95, center_y),
            xytext=(matrix_x + matrix_w + 0.1, center_y),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))

# all_gather label
ax.text(matrix_x + matrix_w + 2.0, center_y + 0.55,
        'all_gather', fontsize=10, ha='center', color='#2E7D32', fontweight='bold')

ax.set_title('RowwiseParallel 词嵌入层', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-0.2, matrix_x + matrix_w + 6.8)
ax.set_ylim(1.0, matrix_y + matrix_h + 1.5)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_embedding_rowwise.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_embedding_rowwise.svg')

# Figure 6: Attention q/k/v ColwiseParallel
fig, ax = plt.subplots(figsize=(12, 6))

gpu_colors = ['#ffccbc', '#ffab91', '#ff8a65', '#ff7043']
gpu_edge = '#d84315'
attn_color = '#fff3e0'
attn_edge = '#ef6c00'

# Input
input_rect = patches.FancyBboxPatch((0.5, 3.5), 2.0, 1.0,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#e3f2fd', edgecolor='#1565c0', linewidth=2)
ax.add_patch(input_rect)
ax.text(1.5, 4.0, '输入 x\n[batch, seq, hidden]\n(Replicate)', fontsize=10, ha='center', va='center')

# 4 GPUs with q/k/v
for i in range(4):
    x = 3.5 + i * 2.2
    # GPU box
    gpu_rect = patches.FancyBboxPatch((x, 2.5), 1.8, 3.0,
                                       boxstyle="round,pad=0.05",
                                       facecolor=gpu_colors[i], edgecolor=gpu_edge, linewidth=2)
    ax.add_patch(gpu_rect)

    ranges = ['[0:H/4)', '[H/4:H/2)', '[H/2:3H/4)', '[3H/4:H)']
    ax.text(x + 0.9, 5.2, f'GPU {i}', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(x + 0.9, 4.6, 'wq/wk/wv', fontsize=9, ha='center', va='center')
    ax.text(x + 0.9, 4.1, 'ColwiseParallel', fontsize=8, ha='center', va='center', color='#d84315')
    ax.text(x + 0.9, 3.3, f'q/k/v\n{ranges[i]}', fontsize=9, ha='center', va='center')

    # Local attention
    attn_rect = patches.FancyBboxPatch((x, 0.5), 1.8, 1.5,
                                        boxstyle="round,pad=0.05",
                                        facecolor=attn_color, edgecolor=attn_edge, linewidth=2)
    ax.add_patch(attn_rect)
    ax.text(x + 0.9, 1.25, f'本地 Attention\nheads {ranges[i]}', fontsize=8, ha='center', va='center')

    # Arrows
    ax.annotate('', xy=(x + 0.9, 5.4), xytext=(2.5, 4.0),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.5,
                               connectionstyle="arc3,rad=0.1"))
    ax.annotate('', xy=(x + 0.9, 2.0), xytext=(x + 0.9, 2.45),
                arrowprops=dict(arrowstyle='->', color=attn_edge, lw=1.5))

ax.set_title('Attention q/k/v: ColwiseParallel', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_attention_qkv_colwise.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_attention_qkv_colwise.svg')

# Figure 7: Attention wo RowwiseParallel
fig, ax = plt.subplots(figsize=(12, 5))

# Heads output shards
for i in range(4):
    x = 0.5 + i * 2.5
    head_rect = patches.FancyBboxPatch((x, 3.0), 1.8, 1.2,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#fff3e0', edgecolor='#ef6c00', linewidth=2)
    ax.add_patch(head_rect)
    ranges = ['[0:H/4)', '[H/4:H/2)', '[H/2:3H/4)', '[3H/4:H)']
    ax.text(x + 0.9, 3.6, f'GPU {i}\nheads {ranges[i]}\n[B,S,hidden/tp]',
            fontsize=8, ha='center', va='center')

    # wo row-sharded
    wo_rect = patches.FancyBboxPatch((x, 1.3), 1.8, 1.2,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(wo_rect)
    ax.text(x + 0.9, 1.9, f'wo\nRowwiseParallel\n输出分片 {i}',
            fontsize=8, ha='center', va='center')

    # Arrows
    ax.annotate('', xy=(x + 0.9, 2.55), xytext=(x + 0.9, 2.95),
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.5))

# all_reduce
comm_rect = patches.FancyBboxPatch((9.5, 2.0), 2.0, 1.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#e1f5fe', edgecolor='#1565c0', linewidth=2)
ax.add_patch(comm_rect)
ax.text(10.5, 2.75, 'all_reduce\n跨 GPU 求和', fontsize=10, ha='center', va='center')

# Output
output_rect = patches.FancyBboxPatch((9.5, 0.2), 2.0, 1.2,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
ax.add_patch(output_rect)
ax.text(10.5, 0.8, '输出\n[batch, seq, hidden]\n(Replicate)', fontsize=9, ha='center', va='center')

# Arrows to all_reduce
for i in range(4):
    x = 0.5 + i * 2.5 + 0.9
    ax.annotate('', xy=(9.5, 2.75), xytext=(x + 0.95, 1.9),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.5,
                               connectionstyle="arc3,rad=0.1"))

ax.annotate('', xy=(10.5, 1.45), xytext=(10.5, 1.95),
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))

ax.set_title('Attention wo: RowwiseParallel', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-0.2, 12.5)
ax.set_ylim(-0.3, 5)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_attention_wo_rowwise.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_attention_wo_rowwise.svg')

# Figure 8: MLP w1/w3 ColwiseParallel
fig, ax = plt.subplots(figsize=(12, 6))

# Input
input_rect = patches.FancyBboxPatch((0.5, 4.0), 2.0, 1.0,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#e3f2fd', edgecolor='#1565c0', linewidth=2)
ax.add_patch(input_rect)
ax.text(1.5, 4.5, '输入 x\n[batch, seq, hidden]\n(Replicate)', fontsize=10, ha='center', va='center')

# w1 and w3 matrices
for i in range(4):
    x = 3.5 + i * 2.0
    ranges = ['[0:I/4)', '[I/4:I/2)', '[I/2:3I/4)', '[3I/4:I)']

    # w1
    w1_rect = patches.FancyBboxPatch((x, 5.5), 1.6, 1.0,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#ffccbc', edgecolor='#d84315', linewidth=1.5)
    ax.add_patch(w1_rect)
    ax.text(x + 0.8, 6.0, f'w1\n列 {ranges[i]}', fontsize=8, ha='center', va='center')

    # w3
    w3_rect = patches.FancyBboxPatch((x, 3.5), 1.6, 1.0,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#ffab91', edgecolor='#d84315', linewidth=1.5)
    ax.add_patch(w3_rect)
    ax.text(x + 0.8, 4.0, f'w3\n列 {ranges[i]}', fontsize=8, ha='center', va='center')

    # SwiGLU
    swiglu_rect = patches.FancyBboxPatch((x, 1.0), 1.6, 1.5,
                                          boxstyle="round,pad=0.05",
                                          facecolor='#fff3e0', edgecolor='#ef6c00', linewidth=2)
    ax.add_patch(swiglu_rect)
    ax.text(x + 0.8, 1.75, f'GPU {i}\nsilu(aᵢ)·bᵢ', fontsize=8, ha='center', va='center')

    # Arrows
    ax.annotate('', xy=(x + 0.8, 6.45), xytext=(2.5, 5.0),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.2))
    ax.annotate('', xy=(x + 0.8, 4.55), xytext=(2.5, 4.7),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.2))
    ax.annotate('', xy=(x + 0.8, 2.55), xytext=(x + 0.8, 3.4),
                arrowprops=dict(arrowstyle='->', color='#ef6c00', lw=1.5))
    ax.annotate('', xy=(x + 0.8, 2.55), xytext=(x + 0.8, 5.45),
                arrowprops=dict(arrowstyle='->', color='#ef6c00', lw=1.5))

ax.text(5.5, 6.8, 'w1 / w3: ColwiseParallel', fontsize=12, ha='center', fontweight='bold', color='#d84315')
ax.set_title('MLP w1/w3: ColwiseParallel', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 12)
ax.set_ylim(0.3, 7.5)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_mlp_w1w3_colwise.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_mlp_w1w3_colwise.svg')

# Figure 9: MLP w2 RowwiseParallel
fig, ax = plt.subplots(figsize=(12, 5))

# SwiGLU output shards
for i in range(4):
    x = 0.5 + i * 2.5
    ranges = ['[0:I/4)', '[I/4:I/2)', '[I/2:3I/4)', '[3I/4:I)']

    swiglu_rect = patches.FancyBboxPatch((x, 3.0), 1.8, 1.2,
                                          boxstyle="round,pad=0.05",
                                          facecolor='#fff3e0', edgecolor='#ef6c00', linewidth=2)
    ax.add_patch(swiglu_rect)
    ax.text(x + 0.9, 3.6, f'GPU {i}\nintermediate\n{ranges[i]}\n[B,S,I/tp]',
            fontsize=8, ha='center', va='center')

    # w2 row
    w2_rect = patches.FancyBboxPatch((x, 1.3), 1.8, 1.2,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(w2_rect)
    ax.text(x + 0.9, 1.9, f'w2\n行 {ranges[i]}', fontsize=8, ha='center', va='center')

    ax.annotate('', xy=(x + 0.9, 2.55), xytext=(x + 0.9, 2.95),
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.5))

# all_reduce
comm_rect = patches.FancyBboxPatch((9.5, 2.0), 2.0, 1.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#e1f5fe', edgecolor='#1565c0', linewidth=2)
ax.add_patch(comm_rect)
ax.text(10.5, 2.75, 'all_reduce\n跨 GPU 求和', fontsize=10, ha='center', va='center')

# Output
output_rect = patches.FancyBboxPatch((9.5, 0.2), 2.0, 1.2,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
ax.add_patch(output_rect)
ax.text(10.5, 0.8, '输出\n[batch, seq, hidden]\n(Replicate)', fontsize=9, ha='center', va='center')

for i in range(4):
    x = 0.5 + i * 2.5 + 0.9
    ax.annotate('', xy=(9.5, 2.75), xytext=(x + 0.95, 1.9),
                arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.5,
                               connectionstyle="arc3,rad=0.1"))

ax.annotate('', xy=(10.5, 1.45), xytext=(10.5, 1.95),
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))

ax.set_title('MLP w2: RowwiseParallel', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-0.2, 12.5)
ax.set_ylim(-0.3, 5)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_mlp_w2_rowwise.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_mlp_w2_rowwise.svg')

# Figure 10: SequenceParallel
fig, ax = plt.subplots(figsize=(12, 5))

# Full sequence
seq_colors = ['#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6']
for i in range(4):
    rect = patches.FancyBboxPatch((1.5 + i * 1.2, 4.0), 1.0, 0.8,
                                   boxstyle="round,pad=0.03",
                                   facecolor=seq_colors[i], edgecolor='#1565c0', linewidth=1.5)
    ax.add_patch(rect)
    labels = ['t₀', 't₁', '...', 'tₛ₋₁']
    ax.text(2.0 + i * 1.2, 4.4, labels[i], fontsize=10, ha='center', va='center')

ax.text(3.9, 5.0, '完整 token 序列 (Replicate)\n[batch, seq, hidden]', fontsize=11, ha='center', fontweight='bold')

# Split to GPUs
for i in range(4):
    x = 1.5 + i * 2.2
    rect = patches.FancyBboxPatch((x, 1.8), 1.8, 1.5,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#e0f7fa', edgecolor='#00838f', linewidth=2)
    ax.add_patch(rect)
    ranges = ['t₀ ~ tₛ/₄₋₁', 'tₛ/₄ ~ tₛ/₂₋₁', 'tₛ/₂ ~ t₃ₛ/₄₋₁', 't₃ₛ/₄ ~ tₛ₋₁']
    ax.text(x + 0.9, 2.55, f'GPU {i}\n{ranges[i]}\n[B, S/tp, H]',
            fontsize=8, ha='center', va='center')

# Output shards
for i in range(4):
    x = 1.5 + i * 2.2
    rect = patches.FancyBboxPatch((x, 0.2), 1.8, 1.0,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 0.9, 0.7, f'Shard(1)\nGPU {i}', fontsize=8, ha='center', va='center')

# Arrows
for i in range(4):
    x = 2.0 + i * 1.2
    target_x = 1.5 + i * 2.2 + 0.9
    ax.annotate('', xy=(target_x, 3.35), xytext=(x, 4.0),
                arrowprops={'arrowstyle': '->', 'color': '#00838f', 'lw': 1.5,
                           'connectionstyle': 'arc3,rad=0.1'})
    ax.annotate('', xy=(target_x, 1.75), xytext=(target_x, 1.75),
                arrowprops={'arrowstyle': '->', 'color': '#2e7d32', 'lw': 1.5})

ax.text(6.0, 1.4, 'SequenceParallel\n按序列维度 Shard(1)', fontsize=11, ha='center', fontweight='bold', color='#00838f')
ax.set_title('SequenceParallel：按 token 序列切分', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0.5, 11)
ax.set_ylim(-0.3, 5.8)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_sequence_parallel.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_sequence_parallel.svg')

# Figure 11: SP + Attention/FeedForward bridge
fig, ax = plt.subplots(figsize=(12, 3))

stages = [
    ('attention_norm\nSequenceParallel', 'Shard(1)\n各 GPU 持 1/4 token', '#e0f7fa', '#00838f'),
    ('PrepareModuleInput', 'all_gather\nShard(1) → Replicate', '#f3e5f5', '#6a1b9a'),
    ('Attention / FeedForward', 'ColwiseParallel\n+ RowwiseParallel', '#fff3e0', '#ef6c00'),
    ('输出转换', 'Replicate → Shard(1)', '#e0f7fa', '#00838f'),
]

for i, (title, content, facecolor, edgecolor) in enumerate(stages):
    x = 0.5 + i * 3.0
    rect = patches.FancyBboxPatch((x, 0.5), 2.5, 1.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 1.25, 1.7, title, fontsize=9, ha='center', va='center', fontweight='bold')
    ax.text(x + 1.25, 1.1, content, fontsize=8, ha='center', va='center')
    if i < len(stages) - 1:
        ax.annotate('', xy=(x + 2.6, 1.4), xytext=(x + 2.55, 1.4),
                    arrowprops={'arrowstyle': '->', 'color': '#424242', 'lw': 2})

ax.set_title('序列并行与 Attention/FeedForward 的衔接', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 3)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_sp_attention_bridge.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_sp_attention_bridge.svg')

# Figure 12: TransformerBlock data flow (improved styling)
fig, ax = plt.subplots(figsize=(10, 11.5))

# Soft shadow for the block
shadow = patches.FancyBboxPatch(
    (1.65, 0.85), 7.0, 8.7,
    boxstyle="round,pad=0.05,rounding_size=0.25",
    facecolor='#E0E0E0', edgecolor='none', alpha=0.35)
ax.add_patch(shadow)

# Block outline
block = patches.FancyBboxPatch(
    (1.5, 1.0), 7.0, 8.7,
    boxstyle="round,pad=0.05,rounding_size=0.25",
    facecolor='#FAFAFA', edgecolor='#616161', linewidth=2.5)
ax.add_patch(block)

ax.text(5.0, 9.85, 'TransformerBlock', fontsize=15, ha='center',
        fontweight='bold', color='#424242')
ax.text(5.0, 9.45, '序列并行下数据流', fontsize=10, ha='center',
        color='#757575')

# Component definitions
components = [
    (8.3, 'attention_norm\nSequenceParallel  Shard(1)', '#E0F7FA', '#00838F'),
    (7.3, 'PrepareModuleInput\nShard(1) → Replicate', '#F3E5F5', '#6A1B9A'),
    (6.3, 'Attention\nColwiseParallel + RowwiseParallel', '#FFF3E0', '#EF6C00'),
    (5.1, '残差连接', '#F5F5F5', '#9E9E9E'),
    (4.1, 'ffn_norm\nSequenceParallel  Shard(1)', '#E0F7FA', '#00838F'),
    (3.1, 'PrepareModuleInput\nShard(1) → Replicate', '#F3E5F5', '#6A1B9A'),
    (2.1, 'FeedForward\nColwiseParallel + RowwiseParallel', '#FFF3E0', '#EF6C00'),
    (0.9, '残差连接', '#F5F5F5', '#9E9E9E'),
]

for y, label, face, edge in components:
    rect = patches.FancyBboxPatch(
        (2.5, y), 5.0, 0.7,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor=face, edgecolor=edge, linewidth=2)
    ax.add_patch(rect)
    ax.text(5.0, y + 0.35, label, fontsize=9, ha='center', va='center',
            color='#212121', fontweight='bold')

# Plus circles for residual joins
for y in (5.45, 1.25):
    circle = patches.Circle((5.0, y), 0.28, facecolor='#ECEFF1',
                            edgecolor='#607D8B', linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(5.0, y, '+', fontsize=14, ha='center', va='center',
            color='#607D8B', fontweight='bold', zorder=6)

# Input / Output boxes
input_rect = patches.FancyBboxPatch(
    (3.5, 10.1), 3.0, 0.7,
    boxstyle="round,pad=0.03,rounding_size=0.1",
    facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
ax.add_patch(input_rect)
ax.text(5.0, 10.45, '输入 x\nShard(1)  [B,S/tp,H]',
        fontsize=9, ha='center', va='center', color='#212121', fontweight='bold')

output_rect = patches.FancyBboxPatch(
    (3.5, 0.2), 3.0, 0.7,
    boxstyle="round,pad=0.03,rounding_size=0.1",
    facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(output_rect)
ax.text(5.0, 0.55, '输出\nShard(1)  [B,S/tp,H]',
        fontsize=9, ha='center', va='center', color='#212121', fontweight='bold')

# Main vertical flow arrows (skip residual rows)
flow_ys = [10.1, 8.95, 7.95, 6.95, 5.45, 4.75, 3.75, 2.75, 1.25, 0.55]
for i in range(len(flow_ys) - 1):
    y1 = flow_ys[i]
    y2 = flow_ys[i + 1]
    if y1 == 10.1:
        y1 = 10.1
    else:
        y1 = y1 - 0.35 if y1 not in (5.45, 1.25) else y1 - 0.28
    if y2 == 0.55:
        y2 = 0.55
    else:
        y2 = y2 + 0.35 if y2 not in (5.45, 1.25) else y2 + 0.28
    ax.annotate('', xy=(5.0, y2), xytext=(5.0, y1),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5))

# Residual bypass arrows (right side, smooth curves)
ax.annotate('', xy=(5.28, 5.45), xytext=(6.5, 10.45),
            arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=2,
                            connectionstyle='arc3,rad=0.35',
                            linestyle='--'))
ax.annotate('', xy=(5.28, 1.25), xytext=(6.5, 5.45),
            arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=2,
                            connectionstyle='arc3,rad=0.25',
                            linestyle='--'))

# Small labels for residuals
ax.text(7.35, 7.8, 'x', fontsize=10, color='#90A4AE', fontweight='bold')
ax.text(7.0, 3.1, 'h', fontsize=10, color='#90A4AE', fontweight='bold')

ax.set_title('序列并行下 TransformerBlock 数据流', fontsize=14,
             fontweight='bold', pad=20)
ax.set_xlim(0.5, 9.5)
ax.set_ylim(-0.3, 11.2)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_transformer_block.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_transformer_block.svg')

# Figure 13: Output layer ColwiseParallel
fig, ax = plt.subplots(figsize=(12, 5))

# Input
input_rect = patches.FancyBboxPatch((0.3, 2.8), 2.0, 1.2,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#e3f2fd', edgecolor='#1565c0', linewidth=2)
ax.add_patch(input_rect)
ax.text(1.3, 3.4, '隐藏状态\n[batch, seq, hidden]\n(Replicate)', fontsize=10, ha='center', va='center')

# Weight matrix columns
for i in range(4):
    x = 2.8 + i * 2.0
    ranges = ['[0:V/4)', '[V/4:V/2)', '[V/2:3V/4)', '[3V/4:V)']
    rect = patches.FancyBboxPatch((x, 2.5), 1.6, 1.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#fff9c4', edgecolor='#f9a825', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 0.8, 3.85, f'GPU {i}', fontsize=9, ha='center', va='center', fontweight='bold')
    ax.text(x + 0.8, 3.35, 'output', fontsize=9, ha='center', va='center')
    ax.text(x + 0.8, 2.95, 'ColwiseParallel', fontsize=8, ha='center', va='center', color='#f9a825')
    ax.text(x + 0.8, 2.55, f'词表列 {ranges[i]}', fontsize=8, ha='center', va='center')

# all_gather
comm_rect = patches.FancyBboxPatch((9.5, 2.8), 2.0, 1.2,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#e1f5fe', edgecolor='#1565c0', linewidth=2)
ax.add_patch(comm_rect)
ax.text(10.5, 3.4, 'all_gather\n拼接词表维度', fontsize=10, ha='center', va='center')

# Output
output_rect = patches.FancyBboxPatch((9.5, 0.8), 2.0, 1.2,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
ax.add_patch(output_rect)
ax.text(10.5, 1.4, '完整 logits\n[batch, seq, vocab_size]\n(Replicate)', fontsize=9, ha='center', va='center')

# Arrows
for i in range(4):
    x = 2.8 + i * 2.0 + 0.8
    ax.annotate('', xy=(x, 4.35), xytext=(2.35, 3.4),
                arrowprops={'arrowstyle': '->', 'color': '#1565c0', 'lw': 1.5,
                           'connectionstyle': 'arc3,rad=0.1'})
    ax.annotate('', xy=(9.5, 3.4), xytext=(x + 0.85, 3.4),
                arrowprops={'arrowstyle': '->', 'color': '#f9a825', 'lw': 1.5})

ax.annotate('', xy=(10.5, 2.05), xytext=(10.5, 2.75),
            arrowprops={'arrowstyle': '->', 'color': '#2e7d32', 'lw': 2})

ax.set_title('输出层 output: ColwiseParallel', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-0.2, 12.5)
ax.set_ylim(0.2, 5.2)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_output_colwise.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_output_colwise.svg')

# Figure 14: Output logits all_gather
fig, ax = plt.subplots(figsize=(15, 4.8))

ranges = ['[0:V/4)', '[V/4:V/2)', '[V/2:3V/4)', '[3V/4:V)']

# GPU shards
shard_xs = [0.5 + i * 2.3 for i in range(4)]
for i, x in enumerate(shard_xs):
    rect = patches.FancyBboxPatch((x, 2.5), 1.9, 1.2,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#fff9c4', edgecolor='#f9a825', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 0.95, 3.1, f'GPU {i}\nvocab {ranges[i]}',
            fontsize=9, ha='center', va='center')

# all_gather
comm_x = 10.2
comm_rect = patches.FancyBboxPatch((comm_x, 2.6), 1.9, 1.0,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#e1f5fe', edgecolor='#1565c0', linewidth=2)
ax.add_patch(comm_rect)
ax.text(comm_x + 0.95, 3.1, 'all_gather\n拼接词表分片',
        fontsize=9, ha='center', va='center')

# Full logits per GPU
full_xs = [13.0 + i * 1.6 for i in range(4)]
for i, x in enumerate(full_xs):
    rect = patches.FancyBboxPatch((x, 2.6), 1.3, 1.0,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 0.65, 3.1, f'GPU {i}\n完整 logits',
            fontsize=8, ha='center', va='center')

# Arrows from shards to all_gather (fan into different y positions)
target_ys = [2.85, 2.95, 3.15, 3.25]
for i, x in enumerate(shard_xs):
    ax.annotate('', xy=(comm_x, target_ys[i]), xytext=(x + 1.9, 3.1),
                arrowprops={'arrowstyle': '->', 'color': '#1565c0', 'lw': 1.5,
                           'connectionstyle': 'arc3,rad=0.05'})

# Arrows from all_gather to full logits boxes
for x in full_xs:
    ax.annotate('', xy=(x, 3.1), xytext=(comm_x + 1.9, 3.1),
                arrowprops={'arrowstyle': '->', 'color': '#2e7d32', 'lw': 1.5,
                           'connectionstyle': 'arc3,rad=0.05'})

ax.set_title('输出 logits all_gather：分片 → 完整 Replicate',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-0.2, 20)
ax.set_ylim(1.8, 4.5)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_output_all_gather.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_output_all_gather.svg')

# Figure 15: Loss Parallel
fig, ax = plt.subplots(figsize=(13.5, 5.2))

# Input
input_rect = patches.FancyBboxPatch((0.3, 2.7), 1.8, 1.2,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#e3f2fd', edgecolor='#1565c0', linewidth=2)
ax.add_patch(input_rect)
ax.text(1.2, 3.3, '隐藏状态\n[B,S/tp,H]\nShard(1)', fontsize=9, ha='center', va='center')

# Output weight columns
output_ranges = ['[0:V/2)', '[V/2:V)']
output_xs = [2.8, 5.3]
for i, x in enumerate(output_xs):
    rect = patches.FancyBboxPatch((x, 2.7), 1.8, 1.2,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#fff9c4', edgecolor='#f9a825', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 0.9, 3.3, f'GPU {i}\noutput\n词表列 {output_ranges[i]}',
            fontsize=8, ha='center', va='center')

# Loss parallel boxes
loss_xs = [8.0, 10.5]
for i, x in enumerate(loss_xs):
    rect = patches.FancyBboxPatch((x, 2.5), 2.0, 1.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 1.0, 3.3, f'GPU {i}\n本地 token × 本地词表列\n计算交叉熵',
            fontsize=8, ha='center', va='center')

# Arrows: input -> output columns
ax.annotate('', xy=(2.75, 3.3), xytext=(2.15, 3.3),
            arrowprops={'arrowstyle': '->', 'color': '#1565c0', 'lw': 1.5})
ax.annotate('', xy=(5.25, 3.3), xytext=(2.15, 3.3),
            arrowprops={'arrowstyle': '->', 'color': '#1565c0', 'lw': 1.5,
                       'connectionstyle': 'arc3,rad=0.15'})

# Arrows: output columns -> loss boxes
ax.annotate('', xy=(7.95, 3.3), xytext=(4.65, 3.3),
            arrowprops={'arrowstyle': '->', 'color': '#f9a825', 'lw': 1.5})
ax.annotate('', xy=(10.45, 3.3), xytext=(7.15, 3.3),
            arrowprops={'arrowstyle': '->', 'color': '#f9a825', 'lw': 1.5})

# loss_parallel bracket (manual to avoid overlap)
bracket_y = 4.35
ax.plot([7.8, 7.8], [bracket_y, bracket_y + 0.25], color='#2e7d32', lw=2)
ax.plot([12.6, 12.6], [bracket_y, bracket_y + 0.25], color='#2e7d32', lw=2)
ax.plot([7.8, 12.6], [bracket_y + 0.25, bracket_y + 0.25], color='#2e7d32', lw=2)
ax.text(10.2, bracket_y + 0.45, 'loss_parallel() 上下文',
        fontsize=10, ha='center', color='#2e7d32', fontweight='bold')

ax.set_title('Loss Parallel：分片计算交叉熵，无需 all_gather 完整 logits',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-0.2, 13.5)
ax.set_ylim(2.0, 5.2)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_loss_parallel.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_loss_parallel.svg')

# Figure 16: PrepareModuleInput
fig, ax = plt.subplots(figsize=(12, 4.5))

# Input
input_items = [
    ('激活张量\n[batch, seq, hidden]\nShard(1)', '#e3f2fd', '#1565c0'),
    ('注意力掩码\n[batch, seq, seq]\nReplicate', '#e3f2fd', '#1565c0'),
]
for i, (text, face, edge) in enumerate(input_items):
    rect = patches.FancyBboxPatch((0.3 + i * 2.3, 2.0), 2.0, 1.5,
                                   boxstyle="round,pad=0.05",
                                   facecolor=face, edgecolor=edge, linewidth=2)
    ax.add_patch(rect)
    ax.text(1.3 + i * 2.3, 2.75, text, fontsize=9, ha='center', va='center')

# PrepareModuleInput
prep_rect = patches.FancyBboxPatch((5.4, 1.7), 3.0, 2.0,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#f3e5f5', edgecolor='#6a1b9a', linewidth=2)
ax.add_patch(prep_rect)
ax.text(6.9, 3.0, 'PrepareModuleInput', fontsize=11, ha='center', va='center', fontweight='bold')
ax.text(6.9, 2.3, '激活: all_gather Shard(1)→Replicate\n掩码: 已是 Replicate，无需通信',
        fontsize=8, ha='center', va='center')

# Output
output_items = [
    ('激活张量\n[batch, seq, hidden]\nReplicate', '#e8f5e9', '#2e7d32'),
    ('注意力掩码\n[batch, seq, seq]\nReplicate', '#e8f5e9', '#2e7d32'),
]
for i, (text, face, edge) in enumerate(output_items):
    rect = patches.FancyBboxPatch((9.2 + i * 2.3, 2.0), 2.0, 1.5,
                                   boxstyle="round,pad=0.05",
                                   facecolor=face, edgecolor=edge, linewidth=2)
    ax.add_patch(rect)
    ax.text(10.2 + i * 2.3, 2.75, text, fontsize=9, ha='center', va='center')

# Arrows
ax.annotate('', xy=(5.35, 2.75), xytext=(2.35, 2.75),
            arrowprops={'arrowstyle': '->', 'color': '#1565c0', 'lw': 2})
ax.annotate('', xy=(5.35, 2.75), xytext=(4.65, 2.75),
            arrowprops={'arrowstyle': '->', 'color': '#1565c0', 'lw': 2})
ax.annotate('', xy=(9.15, 2.75), xytext=(8.35, 2.75),
            arrowprops={'arrowstyle': '->', 'color': '#6a1b9a', 'lw': 2})
ax.annotate('', xy=(11.45, 2.75), xytext=(10.65, 2.75),
            arrowprops={'arrowstyle': '->', 'color': '#6a1b9a', 'lw': 2})

ax.set_title('PrepareModuleInput：布局转换桥', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-0.2, 14)
ax.set_ylim(1.2, 4.3)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_prepare_module_input.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_prepare_module_input.svg')

# Figure 17: Baseline TP vs Sequence Parallel
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def draw_flow(ax, title, steps, layout_type):
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Background
    rect = patches.FancyBboxPatch((0.1, 0.1), 0.8, 0.85,
                                   boxstyle="round,pad=0.02",
                                   facecolor='#fafafa', edgecolor='#bdbdbd', linewidth=1.5)
    ax.add_patch(rect)

    colors = {
        'Replicate': '#e3f2fd',
        'Shard(1)': '#e0f7fa',
        'compute': '#fff3e0',
        'prepare': '#f3e5f5',
    }
    edges = {
        'Replicate': '#1565c0',
        'Shard(1)': '#00838f',
        'compute': '#ef6c00',
        'prepare': '#6a1b9a',
    }

    n = len(steps)
    y_positions = np.linspace(0.75, 0.2, n)

    for i, (label, kind) in enumerate(steps):
        y = y_positions[i]
        rect = patches.FancyBboxPatch((0.2, y - 0.05), 0.6, 0.1,
                                       boxstyle="round,pad=0.01",
                                       facecolor=colors[kind], edgecolor=edges[kind], linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y, label, fontsize=9, ha='center', va='center')

        if i < n - 1:
            ax.annotate('', xy=(0.5, y_positions[i + 1] + 0.06), xytext=(0.5, y - 0.06),
                        arrowprops={'arrowstyle': '->', 'color': '#424242', 'lw': 1.5})

# Baseline TP
baseline_steps = [
    ('输入\n[B,S,H] Replicate', 'Replicate'),
    ('Attention / FFN\n内部分片计算', 'compute'),
    ('输出\n[B,S,H] Replicate', 'Replicate'),
]
draw_flow(ax1, '基本张量并行', baseline_steps, 'baseline')

# Sequence Parallel
sp_steps = [
    ('输入\n[B,S/tp,H] Shard(1)', 'Shard(1)'),
    ('RMSNorm\n分片计算', 'Shard(1)'),
    ('PrepareModuleInput\nShard(1) → Replicate', 'prepare'),
    ('Attention / FFN\n内部分片计算', 'compute'),
    ('输出\n[B,S/tp,H] Shard(1)', 'Shard(1)'),
]
draw_flow(ax2, '序列并行', sp_steps, 'sp')

# Evolution arrow between subplots
fig.text(0.5, 0.5, '进化', fontsize=14, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='#fff9c4', edgecolor='#f9a825', linewidth=2))

for ax in [ax1, ax2]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/robin/work_dir/ScaleTorch/docs/images/fsdp2_baseline_vs_sp.svg',
            format='svg', bbox_inches='tight', dpi=150)
plt.close()
print('Saved fsdp2_baseline_vs_sp.svg')
