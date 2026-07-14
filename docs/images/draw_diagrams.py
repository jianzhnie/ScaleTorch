import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = ['Heiti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as editable text in SVG

BASE_DIR = Path(__file__).parent.resolve()

# Color palette (consistent with the tutorial)
COL_REPLICATE = '#e3f2fd'
COL_REPLICATE_EDGE = '#1565c0'
COL_COLWISE = '#ffccbc'
COL_COLWISE_EDGE = '#d84315'
COL_ROWWISE = '#c8e6c9'
COL_ROWWISE_EDGE = '#2e7d32'
COL_SEQ = '#e0f7fa'
COL_SEQ_EDGE = '#00838f'
COL_PREP = '#f3e5f5'
COL_PREP_EDGE = '#6a1b9a'
COL_COMPUTE = '#fff3e0'
COL_COMPUTE_EDGE = '#ef6c00'
COL_COMM = '#e1f5fe'
COL_COMM_EDGE = '#1565c0'
COL_OUTPUT = '#e8f5e9'
COL_OUTPUT_EDGE = '#2e7d32'
COL_EMBED = '#fff9c4'
COL_EMBED_EDGE = '#f9a825'
COL_NEUTRAL = '#fafafa'
COL_NEUTRAL_EDGE = '#616161'
COL_GPU = '#bbdefb'
COL_GPU_EDGE = '#1565c0'


def save(fig, name):
    """Save figure to SVG and close it."""
    path = BASE_DIR / name
    fig.savefig(path, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {name}')


def rounded_box(ax,
                xy,
                width,
                height,
                facecolor,
                edgecolor,
                linewidth=2,
                radius=0.12,
                text='',
                fontsize=10,
                fontweight='bold',
                text_color='#212121',
                ha='center',
                va='center',
                zorder=2):
    """Draw a rounded rectangle with centered text."""
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder)
    ax.add_patch(box)
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2
    ax.text(cx,
            cy,
            text,
            fontsize=fontsize,
            ha=ha,
            va=va,
            fontweight=fontweight,
            color=text_color,
            zorder=zorder + 1)
    return box


def arrow(ax,
          start,
          end,
          color='#424242',
          lw=1.5,
          style='->',
          connectionstyle='arc3,rad=0.0',
          linestyle='solid',
          **kwargs):
    """Draw an arrow between two points."""
    ax.annotate('',
                xy=end,
                xytext=start,
                arrowprops=dict(arrowstyle=style,
                                color=color,
                                lw=lw,
                                connectionstyle=connectionstyle,
                                linestyle=linestyle),
                **kwargs)


# ---------------------------------------------------------------------------
# Figure 1: FSDP2 + TP overview
# ---------------------------------------------------------------------------
def draw_overview():
    fig, ax = plt.subplots(figsize=(14, 2.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.6)
    ax.axis('off')

    boxes = [
        (0.3, 0.65, 2.8, 1.2, COL_REPLICATE, COL_REPLICATE_EDGE,
         '大规模\nTransformer 模型'),
        (3.7, 0.65, 2.8, 1.2, COL_COMPUTE, COL_COMPUTE_EDGE,
         'TP / SP\n8 GPU × NVLink'),
        (7.1, 0.65, 3.0, 1.2, COL_OUTPUT, COL_OUTPUT_EDGE,
         'FSDP2\n参数 / 梯度 / 优化器状态分片'),
        (10.7, 0.65, 2.8, 1.2, COL_PREP, COL_PREP_EDGE, '数百 ~ 数千 GPU'),
    ]
    for x, y, w, h, fc, ec, txt in boxes:
        rounded_box(ax, (x, y), w, h, fc, ec, text=txt, fontsize=11)

    arrow(ax, (3.1, 1.25), (3.7, 1.25), color=COL_REPLICATE_EDGE, lw=2)
    ax.text(3.4,
            1.5,
            '主机内',
            fontsize=9,
            ha='center',
            color=COL_REPLICATE_EDGE,
            fontweight='bold')
    arrow(ax, (6.5, 1.25), (7.1, 1.25), color=COL_COMPUTE_EDGE, lw=2)
    ax.text(6.8,
            1.5,
            '跨主机',
            fontsize=9,
            ha='center',
            color=COL_COMPUTE_EDGE,
            fontweight='bold')
    arrow(ax, (10.1, 1.25), (10.7, 1.25), color=COL_OUTPUT_EDGE, lw=2)

    ax.set_title('FSDP2 + 张量并行训练大规模 Transformer',
                 fontsize=15,
                 fontweight='bold',
                 pad=5)
    save(fig, 'fsdp2_overview.svg')


# ---------------------------------------------------------------------------
# Figure 3: 1-D DeviceMesh
# ---------------------------------------------------------------------------
def draw_device_mesh_1d():
    fig, ax = plt.subplots(figsize=(12, 2.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.0)
    ax.axis('off')

    n = 8
    box_w, box_h = 1.1, 0.9
    start_x = 0.6
    y = 0.55
    gap = 0.15

    # Backbone line
    ax.plot([start_x, start_x + n * (box_w + gap) - gap],
            [y + box_h / 2, y + box_h / 2],
            color=COL_REPLICATE_EDGE,
            linewidth=3,
            alpha=0.5,
            zorder=1)

    for i in range(n):
        x = start_x + i * (box_w + gap)
        rounded_box(ax, (x, y),
                    box_w,
                    box_h,
                    COL_GPU,
                    COL_GPU_EDGE,
                    text=f'GPU {i}',
                    fontsize=10,
                    radius=0.08)

    ax.text(6.0,
            1.72,
            "tp_mesh = init_device_mesh('cuda', (8,))\n一维 DeviceMesh：8 路张量并行",
            fontsize=13,
            ha='center',
            fontweight='bold',
            color='#212121')
    save(fig, 'fsdp2_device_mesh_1d.svg')


# ---------------------------------------------------------------------------
# Figure 4: TransformerBlock ParallelStyle plan
# ---------------------------------------------------------------------------
def draw_transformer_block_plan():
    fig, ax = plt.subplots(figsize=(9, 11.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 11.5)
    ax.axis('off')

    # Title
    ax.text(4.5,
            11.05,
            'TransformerBlock ParallelStyle 选择',
            fontsize=16,
            ha='center',
            fontweight='bold',
            color='#212121')
    ax.text(
        4.5,
        10.7,
        '灰色=SequenceParallel | 橙色=ColwiseParallel | '
        '绿色=RowwiseParallel | 紫色=PrepareModuleInput',
        fontsize=9,
        ha='center',
        color='#616161')

    # Helper to draw stacked component
    def comp(y, label, fc, ec, h=0.65):
        rounded_box(ax, (1.8, y),
                    5.4,
                    h,
                    fc,
                    ec,
                    text=label,
                    fontsize=10,
                    radius=0.1)
        return y + h / 2

    # Vertical flow (compact)
    flow = [
        (10.0, '输入 x\nReplicate', COL_REPLICATE, COL_REPLICATE_EDGE),
        (9.2, 'tok_embeddings\nRowwiseParallel  词表维度行切分', COL_EMBED,
         COL_EMBED_EDGE),
        (8.4, 'attention_norm\nSequenceParallel  Shard(1)', COL_SEQ,
         COL_SEQ_EDGE),
        (7.6, 'PrepareModuleInput\nShard(1) → Replicate', COL_PREP,
         COL_PREP_EDGE),
        (6.6, 'Attention\nColwiseParallel + RowwiseParallel', COL_COMPUTE,
         COL_COMPUTE_EDGE, 0.9),
        (5.5, '残差连接', COL_NEUTRAL, COL_NEUTRAL_EDGE),
        (4.7, 'ffn_norm\nSequenceParallel  Shard(1)', COL_SEQ, COL_SEQ_EDGE),
        (3.9, 'PrepareModuleInput\nShard(1) → Replicate', COL_PREP,
         COL_PREP_EDGE),
        (2.9, 'FeedForward\nColwiseParallel + RowwiseParallel', COL_COMPUTE,
         COL_COMPUTE_EDGE, 0.9),
        (1.8, '残差连接', COL_NEUTRAL, COL_NEUTRAL_EDGE),
        (0.95, 'output\nColwiseParallel  词表维度列切分', COL_EMBED,
         COL_EMBED_EDGE),
    ]

    ys = []
    for item in flow:
        y, label, fc, ec = item[0], item[1], item[2], item[3]
        h = item[4] if len(item) > 4 else 0.65
        comp(y, label, fc, ec, h)
        ys.append(y + h / 2)

    # Down arrows
    for i in range(len(ys) - 1):
        y1 = ys[i] - 0.32
        y2 = ys[i + 1] + 0.32
        arrow(ax, (4.5, y1), (4.5, y2), color='#424242', lw=1.5)

    # Annotation: all_reduce inside Attention / FFN
    ax.text(7.7,
            7.05,
            '内部 all_reduce',
            fontsize=9,
            color=COL_COMPUTE_EDGE,
            fontweight='bold',
            ha='left')
    ax.text(7.7,
            3.35,
            '内部 all_reduce',
            fontsize=9,
            color=COL_COMPUTE_EDGE,
            fontweight='bold',
            ha='left')

    save(fig, 'fsdp2_transformer_block_plan.svg')


# ---------------------------------------------------------------------------
# Figure 5: RowwiseParallel Embedding
# ---------------------------------------------------------------------------
def draw_embedding_rowwise():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    n_rows, n_cols, n_gpus = 16, 8, 4
    rows_per_gpu = n_rows // n_gpus
    cell_w, cell_h = 0.52, 0.34
    matrix_x, matrix_y = 3.0, 1.3
    matrix_w = n_cols * cell_w
    matrix_h = n_rows * cell_h
    gpu_colors = ['#FFF9C4', '#FFF59D', '#FFF176', '#FFEE58']

    # Shadow
    shadow = patches.FancyBboxPatch(
        (matrix_x + 0.08, matrix_y - 0.08),
        matrix_w,
        matrix_h,
        boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor='#E0E0E0',
        edgecolor='none',
        alpha=0.3)
    ax.add_patch(shadow)

    # Matrix cells
    for gpu in range(n_gpus):
        for r in range(rows_per_gpu):
            row = gpu * rows_per_gpu + r
            for c in range(n_cols):
                x = matrix_x + c * cell_w
                y = matrix_y + (n_rows - 1 - row) * cell_h
                rect = patches.Rectangle((x, y),
                                         cell_w,
                                         cell_h,
                                         facecolor=gpu_colors[gpu],
                                         edgecolor=COL_EMBED_EDGE,
                                         linewidth=0.5)
                ax.add_patch(rect)

    # Border
    border = patches.FancyBboxPatch(
        (matrix_x, matrix_y),
        matrix_w,
        matrix_h,
        boxstyle='round,pad=0.02,rounding_size=0.08',
        facecolor='none',
        edgecolor=COL_EMBED_EDGE,
        linewidth=2)
    ax.add_patch(border)

    ax.text(matrix_x + matrix_w / 2,
            matrix_y + matrix_h + 0.45,
            'embedding_weight: [vocab_size, hidden_size]',
            fontsize=12,
            ha='center',
            va='bottom',
            fontweight='bold',
            color='#424242')
    ax.text(matrix_x - 0.6,
            matrix_y + matrix_h / 2,
            '词表维度\n(按行切分)',
            fontsize=10,
            ha='center',
            va='center',
            rotation=90,
            color='#424242')
    ax.text(matrix_x + matrix_w / 2,
            matrix_y - 0.55,
            'hidden_size',
            fontsize=10,
            ha='center',
            va='top',
            color='#424242')

    # GPU labels
    for gpu in range(n_gpus):
        y_mid = matrix_y + matrix_h - (gpu * rows_per_gpu +
                                       rows_per_gpu / 2) * cell_h
        label_x = matrix_x + matrix_w + 0.7
        rounded_box(ax, (label_x, y_mid - 0.3),
                    0.5,
                    0.6,
                    gpu_colors[gpu],
                    COL_EMBED_EDGE,
                    text=str(gpu),
                    fontsize=9)
        ax.text(
            label_x + 0.8,
            y_mid,
            f'GPU {gpu}    词表行 '
            f'[{gpu * n_rows // n_gpus}:{(gpu + 1) * n_rows // n_gpus})',
            fontsize=10,
            va='center',
            ha='left',
            color='#424242',
            fontweight='bold')
        ax.plot([matrix_x + matrix_w, label_x], [y_mid, y_mid],
                color=COL_EMBED_EDGE,
                linewidth=0.8,
                alpha=0.6)

    # Input / output
    center_y = matrix_y + matrix_h / 2
    rounded_box(ax, (0.3, center_y - 0.65),
                2.2,
                1.3,
                COL_REPLICATE,
                COL_REPLICATE_EDGE,
                text='input_ids\n[batch, seq]\nReplicate',
                fontsize=10)
    rounded_box(ax, (matrix_x + matrix_w + 4.3, center_y - 0.65),
                2.4,
                1.3,
                COL_OUTPUT,
                COL_OUTPUT_EDGE,
                text='output\n[batch, seq, hidden]\nReplicate',
                fontsize=10)

    arrow(ax, (2.55, center_y), (matrix_x - 0.05, center_y),
          color=COL_REPLICATE_EDGE,
          lw=2)
    arrow(ax, (matrix_x + matrix_w + 0.05, center_y),
          (matrix_x + matrix_w + 4.25, center_y),
          color=COL_OUTPUT_EDGE,
          lw=2)
    ax.text(matrix_x + matrix_w + 2.2,
            center_y + 0.55,
            'all_gather',
            fontsize=10,
            ha='center',
            color=COL_OUTPUT_EDGE,
            fontweight='bold')

    ax.set_title('RowwiseParallel 词嵌入层',
                 fontsize=15,
                 fontweight='bold',
                 pad=20)
    save(fig, 'fsdp2_embedding_rowwise.svg')


# ---------------------------------------------------------------------------
# Figure 6: Attention q/k/v ColwiseParallel
# ---------------------------------------------------------------------------
def draw_attention_qkv_colwise():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    rounded_box(ax, (0.4, 3.7),
                2.0,
                1.0,
                COL_REPLICATE,
                COL_REPLICATE_EDGE,
                text='输入 x\n[batch, seq, hidden]\nReplicate',
                fontsize=10)

    gpu_colors = ['#ffccbc', '#ffab91', '#ff8a65', '#ff7043']
    ranges = ['[0:H/4)', '[H/4:H/2)', '[H/2:3H/4)', '[3H/4:H)']

    for i in range(4):
        x = 3.0 + i * 2.4
        # GPU group
        rounded_box(ax, (x, 2.6),
                    2.0,
                    2.6,
                    gpu_colors[i],
                    COL_COLWISE_EDGE,
                    text='',
                    radius=0.1)
        ax.text(x + 1.0,
                4.9,
                f'GPU {i}',
                fontsize=10,
                ha='center',
                fontweight='bold',
                color='#212121')
        ax.text(x + 1.0,
                4.35,
                'wq / wk / wv',
                fontsize=9,
                ha='center',
                color='#212121')
        ax.text(x + 1.0,
                3.9,
                'ColwiseParallel',
                fontsize=8,
                ha='center',
                color=COL_COLWISE_EDGE,
                fontweight='bold')
        ax.text(x + 1.0,
                3.2,
                f'q / k / v\n{ranges[i]}',
                fontsize=8,
                ha='center',
                color='#212121')

        # Local attention
        rounded_box(ax, (x, 0.5),
                    2.0,
                    1.6,
                    COL_COMPUTE,
                    COL_COMPUTE_EDGE,
                    text=f'本地 Attention\nheads {ranges[i]}',
                    fontsize=8)

        arrow(ax, (2.45, 4.2), (x + 1.0, 4.55),
              color=COL_REPLICATE_EDGE,
              lw=1.5,
              connectionstyle='arc3,rad=0.08')
        arrow(ax, (x + 1.0, 2.55), (x + 1.0, 2.15),
              color=COL_COMPUTE_EDGE,
              lw=1.5)

    ax.set_title('Attention q/k/v: ColwiseParallel',
                 fontsize=15,
                 fontweight='bold',
                 pad=20)
    save(fig, 'fsdp2_attention_qkv_colwise.svg')


# ---------------------------------------------------------------------------
# Figure 7: Attention wo RowwiseParallel
# ---------------------------------------------------------------------------
def draw_attention_wo_rowwise():
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.8)
    ax.axis('off')

    box_w = 1.8
    gap = 2.4
    ranges = ['[0:H/4)', '[H/4:H/2)', '[H/2:3H/4)', '[3H/4:H)']
    centers = []
    for i in range(4):
        x = 0.4 + i * gap
        rounded_box(ax, (x, 3.0),
                    box_w,
                    1.2,
                    COL_COMPUTE,
                    COL_COMPUTE_EDGE,
                    text=f'GPU {i}\nheads {ranges[i]}\n[B,S,hidden/tp]',
                    fontsize=8)
        rounded_box(ax, (x, 1.2),
                    box_w,
                    1.2,
                    COL_ROWWISE,
                    COL_ROWWISE_EDGE,
                    text=f'wo\nRowwiseParallel\n输出分片 {i}',
                    fontsize=8)
        arrow(ax, (x + box_w / 2, 2.95), (x + box_w / 2, 2.45),
              color=COL_ROWWISE_EDGE,
              lw=1.5)
        centers.append(x + box_w / 2)

    rounded_box(ax, (11.2, 2.0),
                2.0,
                1.5,
                COL_COMM,
                COL_COMM_EDGE,
                text='all_reduce\n跨 GPU 求和',
                fontsize=10)
    rounded_box(ax, (11.2, 0.2),
                2.0,
                1.2,
                COL_OUTPUT,
                COL_OUTPUT_EDGE,
                text='输出\n[batch, seq, hidden]\nReplicate',
                fontsize=9)

    target_ys = [2.45, 2.7, 2.95, 3.2]
    for i, cx in enumerate(centers):
        arrow(ax, (cx + box_w / 2, 1.75), (11.2, target_ys[i]),
              color=COL_COMM_EDGE,
              lw=1.5,
              connectionstyle='arc3,rad=0.08')

    arrow(ax, (12.2, 1.95), (12.2, 1.45), color=COL_OUTPUT_EDGE, lw=2)
    ax.set_title('Attention wo: RowwiseParallel',
                 fontsize=15,
                 fontweight='bold',
                 pad=20)
    save(fig, 'fsdp2_attention_wo_rowwise.svg')


# ---------------------------------------------------------------------------
# Figure 8: MLP w1/w3 ColwiseParallel
# ---------------------------------------------------------------------------
def draw_mlp_w1w3_colwise():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    rounded_box(ax, (0.4, 4.5),
                2.0,
                1.0,
                COL_REPLICATE,
                COL_REPLICATE_EDGE,
                text='输入 x\n[batch, seq, hidden]\nReplicate',
                fontsize=10)

    ranges = ['[0:I/4)', '[I/4:I/2)', '[I/2:3I/4)', '[3I/4:I)']
    for i in range(4):
        x = 3.0 + i * 2.3
        rounded_box(ax, (x, 5.5),
                    1.8,
                    0.9,
                    COL_COLWISE,
                    COL_COLWISE_EDGE,
                    text=f'w1\n列 {ranges[i]}',
                    fontsize=8)
        rounded_box(ax, (x, 3.6),
                    1.8,
                    0.9,
                    '#ffab91',
                    COL_COLWISE_EDGE,
                    text=f'w3\n列 {ranges[i]}',
                    fontsize=8)
        rounded_box(ax, (x, 1.0),
                    1.8,
                    1.8,
                    COL_COMPUTE,
                    COL_COMPUTE_EDGE,
                    text=f'GPU {i}\nsilu(aᵢ)·bᵢ',
                    fontsize=9)

        arrow(ax, (2.45, 5.0), (x + 0.9, 5.5),
              color=COL_REPLICATE_EDGE,
              lw=1.2)
        arrow(ax, (2.45, 5.0), (x + 0.9, 4.55),
              color=COL_REPLICATE_EDGE,
              lw=1.2,
              connectionstyle='arc3,rad=0.05')
        arrow(ax, (x + 0.9, 5.0), (x + 0.9, 2.85),
              color=COL_COMPUTE_EDGE,
              lw=1.2)
        arrow(ax, (x + 0.9, 3.55), (x + 0.9, 2.85),
              color=COL_COMPUTE_EDGE,
              lw=1.2)

    ax.text(0.5,
            6.3,
            'w1 / w3: ColwiseParallel',
            fontsize=13,
            ha='left',
            fontweight='bold',
            color=COL_COLWISE_EDGE)
    ax.set_title('MLP w1/w3: ColwiseParallel',
                 fontsize=15,
                 fontweight='bold',
                 pad=20)
    save(fig, 'fsdp2_mlp_w1w3_colwise.svg')


# ---------------------------------------------------------------------------
# Figure 9: MLP w2 RowwiseParallel
# ---------------------------------------------------------------------------
def draw_mlp_w2_rowwise():
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.8)
    ax.axis('off')

    box_w = 1.8
    gap = 2.4
    ranges = ['[0:I/4)', '[I/4:I/2)', '[I/2:3I/4)', '[3I/4:I)']
    centers = []
    for i in range(4):
        x = 0.4 + i * gap
        rounded_box(ax, (x, 3.0),
                    box_w,
                    1.2,
                    COL_COMPUTE,
                    COL_COMPUTE_EDGE,
                    text=f'GPU {i}\nintermediate\n{ranges[i]}\n[B,S,I/tp]',
                    fontsize=8)
        rounded_box(ax, (x, 1.2),
                    box_w,
                    1.2,
                    COL_ROWWISE,
                    COL_ROWWISE_EDGE,
                    text=f'w2\n行 {ranges[i]}',
                    fontsize=8)
        arrow(ax, (x + box_w / 2, 2.95), (x + box_w / 2, 2.45),
              color=COL_ROWWISE_EDGE,
              lw=1.5)
        centers.append(x + box_w / 2)

    rounded_box(ax, (11.2, 2.0),
                2.0,
                1.5,
                COL_COMM,
                COL_COMM_EDGE,
                text='all_reduce\n跨 GPU 求和',
                fontsize=10)
    rounded_box(ax, (11.2, 0.2),
                2.0,
                1.2,
                COL_OUTPUT,
                COL_OUTPUT_EDGE,
                text='输出\n[batch, seq, hidden]\nReplicate',
                fontsize=9)

    target_ys = [2.45, 2.7, 2.95, 3.2]
    for i, cx in enumerate(centers):
        arrow(ax, (cx + box_w / 2, 1.75), (11.2, target_ys[i]),
              color=COL_COMM_EDGE,
              lw=1.5,
              connectionstyle='arc3,rad=0.08')

    arrow(ax, (12.2, 1.95), (12.2, 1.45), color=COL_OUTPUT_EDGE, lw=2)
    ax.set_title('MLP w2: RowwiseParallel',
                 fontsize=15,
                 fontweight='bold',
                 pad=20)
    save(fig, 'fsdp2_mlp_w2_rowwise.svg')


# ---------------------------------------------------------------------------
# Figure 10: SequenceParallel
# ---------------------------------------------------------------------------
def draw_sequence_parallel():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    seq_colors = ['#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6']
    for i in range(4):
        rounded_box(ax, (1.5 + i * 1.2, 4.0),
                    1.0,
                    0.75,
                    seq_colors[i],
                    COL_REPLICATE_EDGE,
                    text=['t₀', 't₁', '...', 'tₛ₋₁'][i],
                    fontsize=10)
    ax.text(3.9,
            4.95,
            '完整 token 序列 (Replicate)\n[batch, seq, hidden]',
            fontsize=11,
            ha='center',
            fontweight='bold')

    ranges = ['t₀ ~ tₛ/₄₋₁', 'tₛ/₄ ~ tₛ/₂₋₁', 'tₛ/₂ ~ t₃ₛ/₄₋₁', 't₃ₛ/₄ ~ tₛ₋₁']
    for i in range(4):
        x = 1.5 + i * 2.6
        rounded_box(ax, (x, 1.9),
                    2.2,
                    1.5,
                    COL_SEQ,
                    COL_SEQ_EDGE,
                    text=f'GPU {i}\n{ranges[i]}\n[B, S/tp, H]',
                    fontsize=9)
        rounded_box(ax, (x, 0.4),
                    2.2,
                    1.0,
                    COL_OUTPUT,
                    COL_OUTPUT_EDGE,
                    text=f'Shard(1)\nGPU {i}',
                    fontsize=9)
        arrow(ax, (2.0 + i * 1.2, 4.0), (x + 1.1, 3.45),
              color=COL_SEQ_EDGE,
              lw=1.5,
              connectionstyle='arc3,rad=0.1')
        arrow(ax, (x + 1.1, 1.85), (x + 1.1, 1.45),
              color=COL_OUTPUT_EDGE,
              lw=1.5)

    ax.text(6.8,
            1.4,
            'SequenceParallel\n按序列维度 Shard(1)',
            fontsize=11,
            ha='center',
            fontweight='bold',
            color=COL_SEQ_EDGE)
    ax.set_title('SequenceParallel：按 token 序列切分',
                 fontsize=15,
                 fontweight='bold',
                 pad=20)
    save(fig, 'fsdp2_sequence_parallel.svg')


# ---------------------------------------------------------------------------
# Figure 11: SP + Attention/FeedForward bridge
# ---------------------------------------------------------------------------
def draw_sp_attention_bridge():
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 2.8)
    ax.axis('off')

    stages = [
        ('attention_norm\nSequenceParallel', 'Shard(1)\n各 GPU 持 1/4 token',
         COL_SEQ, COL_SEQ_EDGE),
        ('PrepareModuleInput', 'all_gather\nShard(1) → Replicate', COL_PREP,
         COL_PREP_EDGE),
        ('Attention / FeedForward', 'ColwiseParallel\n+ RowwiseParallel',
         COL_COMPUTE, COL_COMPUTE_EDGE),
        ('输出转换', 'Replicate → Shard(1)', COL_SEQ, COL_SEQ_EDGE),
    ]

    for i, (title, content, fc, ec) in enumerate(stages):
        x = 0.6 + i * 3.1
        rounded_box(ax, (x, 0.4), 2.7, 1.6, fc, ec, text='', radius=0.12)
        ax.text(x + 1.35,
                1.7,
                title,
                fontsize=9,
                ha='center',
                va='center',
                fontweight='bold',
                color='#212121')
        ax.text(x + 1.35,
                0.95,
                content,
                fontsize=8,
                ha='center',
                va='center',
                color='#424242')
        if i < len(stages) - 1:
            arrow(ax, (x + 2.75, 1.2), (x + 3.05, 1.2), color='#424242', lw=2)

    ax.set_title('序列并行与 Attention/FeedForward 的衔接',
                 fontsize=15,
                 fontweight='bold',
                 pad=6)
    save(fig, 'fsdp2_sp_attention_bridge.svg')


# ---------------------------------------------------------------------------
# Figure 12: TransformerBlock data flow
# ---------------------------------------------------------------------------
def draw_transformer_block():
    fig, ax = plt.subplots(figsize=(8, 10.2))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10.2)
    ax.axis('off')

    # Block background
    shadow = patches.FancyBboxPatch(
        (1.6, 0.85),
        5.2,
        7.15,
        boxstyle='round,pad=0.05,rounding_size=0.25',
        facecolor='#E0E0E0',
        edgecolor='none',
        alpha=0.35)
    ax.add_patch(shadow)

    block = patches.FancyBboxPatch(
        (1.5, 1.0),
        5.2,
        7.15,
        boxstyle='round,pad=0.05,rounding_size=0.25',
        facecolor='#FAFAFA',
        edgecolor=COL_NEUTRAL_EDGE,
        linewidth=2.5)
    ax.add_patch(block)

    ax.text(4.0,
            8.65,
            'TransformerBlock',
            fontsize=14,
            ha='center',
            fontweight='bold',
            color='#424242')
    ax.text(4.0, 8.35, '序列并行下数据流', fontsize=9, ha='center', color='#757575')

    components = [
        (7.35, 'attention_norm\nSequenceParallel  Shard(1)', COL_SEQ,
         COL_SEQ_EDGE),
        (6.65, 'PrepareModuleInput\nShard(1) → Replicate', COL_PREP,
         COL_PREP_EDGE),
        (5.85, 'Attention\nColwiseParallel + RowwiseParallel', COL_COMPUTE,
         COL_COMPUTE_EDGE),
        (5.05, '残差连接', COL_NEUTRAL, COL_NEUTRAL_EDGE),
        (4.35, 'ffn_norm\nSequenceParallel  Shard(1)', COL_SEQ, COL_SEQ_EDGE),
        (3.65, 'PrepareModuleInput\nShard(1) → Replicate', COL_PREP,
         COL_PREP_EDGE),
        (2.85, 'FeedForward\nColwiseParallel + RowwiseParallel', COL_COMPUTE,
         COL_COMPUTE_EDGE),
        (2.05, '残差连接', COL_NEUTRAL, COL_NEUTRAL_EDGE),
    ]

    for y, label, fc, ec in components:
        rounded_box(ax, (2.0, y),
                    4.0,
                    0.62,
                    fc,
                    ec,
                    text=label,
                    fontsize=9,
                    radius=0.1)

    # Plus circles
    for y in (5.36, 2.36):
        circle = patches.Circle((4.0, y),
                                0.25,
                                facecolor='#ECEFF1',
                                edgecolor='#607D8B',
                                linewidth=2,
                                zorder=5)
        ax.add_patch(circle)
        ax.text(4.0,
                y,
                '+',
                fontsize=13,
                ha='center',
                va='center',
                color='#607D8B',
                fontweight='bold',
                zorder=6)

    # Input / output
    rounded_box(ax, (2.7, 8.95),
                2.6,
                0.6,
                COL_REPLICATE,
                COL_REPLICATE_EDGE,
                text='输入 x\nShard(1)  [B,S/tp,H]',
                fontsize=9)
    rounded_box(ax, (2.7, 0.25),
                2.6,
                0.6,
                COL_OUTPUT,
                COL_OUTPUT_EDGE,
                text='输出\nShard(1)  [B,S/tp,H]',
                fontsize=9)

    # Vertical flow arrows
    flow_ys = [8.95, 7.68, 6.98, 6.16, 5.36, 4.66, 3.96, 3.16, 2.36, 0.58]
    for i in range(len(flow_ys) - 1):
        y1 = flow_ys[i] - (0.33 if flow_ys[i] not in (5.36, 2.36) else 0.25)
        y2 = flow_ys[i + 1] + (0.33 if flow_ys[i + 1] not in (5.36,
                                                              2.36) else 0.25)
        arrow(ax, (4.0, y1), (4.0, y2), color='#424242', lw=1.5)

    # Residual bypass arrows (run outside the block on the right)
    ax.annotate('',
                xy=(4.25, 5.36),
                xytext=(6.5, 9.18),
                arrowprops=dict(arrowstyle='->',
                                color='#90A4AE',
                                lw=2,
                                connectionstyle='arc3,rad=-0.45',
                                linestyle='--'))
    ax.annotate('',
                xy=(4.25, 2.36),
                xytext=(6.3, 5.36),
                arrowprops=dict(arrowstyle='->',
                                color='#90A4AE',
                                lw=2,
                                connectionstyle='arc3,rad=-0.35',
                                linestyle='--'))
    ax.text(6.8, 7.2, 'x', fontsize=10, color='#90A4AE', fontweight='bold')
    ax.text(6.6, 3.7, 'h', fontsize=10, color='#90A4AE', fontweight='bold')

    ax.set_title('序列并行下 TransformerBlock 数据流',
                 fontsize=14,
                 fontweight='bold',
                 pad=15)
    save(fig, 'fsdp2_transformer_block.svg')


# ---------------------------------------------------------------------------
# Figure 13: Output layer ColwiseParallel
# ---------------------------------------------------------------------------
def draw_output_colwise():
    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    rounded_box(ax, (0.3, 2.5),
                2.0,
                1.1,
                COL_REPLICATE,
                COL_REPLICATE_EDGE,
                text='隐藏状态\n[batch, seq, hidden]\n(Replicate)',
                fontsize=10)

    box_w = 1.7
    gap = 2.2
    ranges = ['[0:V/4)', '[V/4:V/2)', '[V/2:3V/4)', '[3V/4:V)']
    centers = []
    for i in range(4):
        x = 2.8 + i * gap
        rounded_box(ax, (x, 2.1),
                    box_w,
                    1.8,
                    COL_EMBED,
                    COL_EMBED_EDGE,
                    text='')
        ax.text(x + box_w / 2,
                3.55,
                f'GPU {i}',
                fontsize=9,
                ha='center',
                fontweight='bold',
                color='#212121')
        ax.text(x + box_w / 2,
                3.05,
                'output',
                fontsize=9,
                ha='center',
                color='#212121')
        ax.text(x + box_w / 2,
                2.65,
                'ColwiseParallel',
                fontsize=8,
                ha='center',
                color=COL_EMBED_EDGE,
                fontweight='bold')
        ax.text(x + box_w / 2,
                2.25,
                f'词表列 {ranges[i]}',
                fontsize=8,
                ha='center',
                color='#212121')
        centers.append(x + box_w / 2)

    rounded_box(ax, (12.5, 2.5),
                2.0,
                1.1,
                COL_COMM,
                COL_COMM_EDGE,
                text='all_gather\n拼接词表维度',
                fontsize=10)
    rounded_box(ax, (12.5, 0.65),
                2.0,
                1.1,
                COL_OUTPUT,
                COL_OUTPUT_EDGE,
                text='完整 logits\n[batch, seq, vocab_size]\n(Replicate)',
                fontsize=9)

    for i, cx in enumerate(centers):
        # input -> each output column (horizontal to left edge)
        arrow(ax, (2.35, 3.05), (cx - box_w / 2, 3.05),
              color=COL_REPLICATE_EDGE,
              lw=1.5)
        # each output column -> all_gather (fan-in on the right)
        target_y = 2.65 + i * 0.17
        arrow(ax, (cx + box_w / 2, 3.05), (12.5, target_y),
              color=COL_EMBED_EDGE,
              lw=1.5,
              connectionstyle='arc3,rad=0.05')

    arrow(ax, (13.5, 2.45), (13.5, 1.8), color=COL_OUTPUT_EDGE, lw=2)
    ax.set_title('输出层 output: ColwiseParallel',
                 fontsize=15,
                 fontweight='bold',
                 pad=6)
    save(fig, 'fsdp2_output_colwise.svg')


# ---------------------------------------------------------------------------
# Figure 14: Output logits all_gather
# ---------------------------------------------------------------------------
def draw_output_all_gather():
    fig, ax = plt.subplots(figsize=(17, 2.8))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 2.8)
    ax.axis('off')

    ranges = ['[0:V/4)', '[V/4:V/2)', '[V/2:3V/4)', '[3V/4:V)']
    shard_xs = [0.4 + i * 2.2 for i in range(4)]
    for i, x in enumerate(shard_xs):
        rounded_box(ax, (x, 0.9),
                    1.8,
                    0.9,
                    COL_EMBED,
                    COL_EMBED_EDGE,
                    text=f'GPU {i}\nvocab {ranges[i]}',
                    fontsize=9)

    rounded_box(ax, (9.8, 0.9),
                2.0,
                0.9,
                COL_COMM,
                COL_COMM_EDGE,
                text='all_gather\n拼接词表分片',
                fontsize=9)

    full_xs = [12.0 + i * 1.2 for i in range(4)]
    for i, x in enumerate(full_xs):
        rounded_box(ax, (x, 0.9),
                    1.2,
                    0.9,
                    COL_OUTPUT,
                    COL_OUTPUT_EDGE,
                    text=f'GPU {i}\n完整 logits',
                    fontsize=8)

    target_ys = [1.1, 1.2, 1.3, 1.4]
    for i, x in enumerate(shard_xs):
        arrow(ax, (x + 1.8, 1.35), (9.8, target_ys[i]),
              color=COL_COMM_EDGE,
              lw=1.5,
              connectionstyle='arc3,rad=0.05')

    for x in full_xs:
        arrow(ax, (11.8, 1.35), (x, 1.35),
              color=COL_OUTPUT_EDGE,
              lw=1.5,
              connectionstyle='arc3,rad=0.05')

    ax.set_title('输出 logits all_gather：分片 → 完整 Replicate',
                 fontsize=15,
                 fontweight='bold',
                 pad=4)
    save(fig, 'fsdp2_output_all_gather.svg')


# ---------------------------------------------------------------------------
# Figure 15: Loss Parallel
# ---------------------------------------------------------------------------
def draw_loss_parallel():
    fig, ax = plt.subplots(figsize=(13.5, 3.0))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 3.0)
    ax.axis('off')

    rounded_box(ax, (0.3, 0.8),
                1.8,
                1.1,
                COL_REPLICATE,
                COL_REPLICATE_EDGE,
                text='隐藏状态\n[B,S/tp,H]\nShard(1)',
                fontsize=9)

    output_ranges = ['[0:V/2)', '[V/2:V)']
    output_xs = [2.6, 5.0]
    for i, x in enumerate(output_xs):
        rounded_box(ax, (x, 0.8),
                    1.9,
                    1.1,
                    COL_EMBED,
                    COL_EMBED_EDGE,
                    text=f'GPU {i}\noutput\n词表列 {output_ranges[i]}',
                    fontsize=8)

    loss_xs = [7.6, 10.2]
    for i, x in enumerate(loss_xs):
        rounded_box(ax, (x, 0.65),
                    2.1,
                    1.4,
                    COL_OUTPUT,
                    COL_OUTPUT_EDGE,
                    text=f'GPU {i}\n本地 token × 本地词表列\n计算交叉熵',
                    fontsize=8)

    arrow(ax, (2.15, 1.35), (2.6, 1.35), color=COL_REPLICATE_EDGE, lw=1.5)
    arrow(ax, (2.15, 1.35), (5.0, 1.35),
          color=COL_REPLICATE_EDGE,
          lw=1.5,
          connectionstyle='arc3,rad=0.15')
    arrow(ax, (4.55, 1.35), (7.6, 1.35), color=COL_EMBED_EDGE, lw=1.5)
    arrow(ax, (6.95, 1.35), (10.2, 1.35), color=COL_EMBED_EDGE, lw=1.5)

    # Bracket
    bracket_y = 2.15
    ax.plot([7.5, 7.5], [bracket_y, bracket_y + 0.2],
            color=COL_OUTPUT_EDGE,
            lw=2)
    ax.plot([12.35, 12.35], [bracket_y, bracket_y + 0.2],
            color=COL_OUTPUT_EDGE,
            lw=2)
    ax.plot([7.5, 12.35], [bracket_y + 0.2, bracket_y + 0.2],
            color=COL_OUTPUT_EDGE,
            lw=2)
    ax.text(9.9,
            bracket_y + 0.38,
            'loss_parallel() 上下文',
            fontsize=10,
            ha='center',
            color=COL_OUTPUT_EDGE,
            fontweight='bold')

    ax.set_title('Loss Parallel：分片计算交叉熵，无需 all_gather 完整 logits',
                 fontsize=15,
                 fontweight='bold',
                 pad=6)
    save(fig, 'fsdp2_loss_parallel.svg')


# ---------------------------------------------------------------------------
# Figure 16: PrepareModuleInput
# ---------------------------------------------------------------------------
def draw_prepare_module_input():
    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    y_top = 2.55
    y_bot = 0.65

    # Input boxes (stacked vertically)
    rounded_box(ax, (0.3, y_top),
                2.2,
                1.2,
                COL_REPLICATE,
                COL_REPLICATE_EDGE,
                text='激活张量\n[batch, seq, hidden]\nShard(1)',
                fontsize=9)
    rounded_box(ax, (0.3, y_bot),
                2.2,
                1.2,
                COL_REPLICATE,
                COL_REPLICATE_EDGE,
                text='注意力掩码\n[batch, seq, seq]\nReplicate',
                fontsize=9)

    # PrepareModuleInput
    rounded_box(ax, (3.3, 0.35), 3.4, 3.0, COL_PREP, COL_PREP_EDGE, text='')
    ax.text(5.0,
            2.9,
            'PrepareModuleInput',
            fontsize=12,
            ha='center',
            fontweight='bold',
            color='#212121')
    ax.text(5.0,
            1.85,
            '激活: all_gather\nShard(1) → Replicate\n掩码: 已是 Replicate，无需通信',
            fontsize=9,
            ha='center',
            color='#424242')

    # Output boxes
    rounded_box(ax, (7.4, y_top),
                2.2,
                1.2,
                COL_OUTPUT,
                COL_OUTPUT_EDGE,
                text='激活张量\n[batch, seq, hidden]\nReplicate',
                fontsize=9)
    rounded_box(ax, (7.4, y_bot),
                2.2,
                1.2,
                COL_OUTPUT,
                COL_OUTPUT_EDGE,
                text='注意力掩码\n[batch, seq, seq]\nReplicate',
                fontsize=9)

    # Activation path arrows
    arrow(ax, (2.55, 3.15), (3.3, 3.15), color=COL_REPLICATE_EDGE, lw=2)
    ax.text(2.95,
            3.5,
            'all_gather',
            fontsize=8,
            ha='center',
            color=COL_REPLICATE_EDGE,
            fontweight='bold')
    arrow(ax, (6.75, 3.15), (7.4, 3.15), color=COL_PREP_EDGE, lw=2)

    # Mask path arrows (pass-through)
    arrow(ax, (2.55, 1.25), (3.3, 1.25),
          color=COL_REPLICATE_EDGE,
          lw=2,
          linestyle='dashed')
    ax.text(2.95,
            0.9,
            'pass-through',
            fontsize=8,
            ha='center',
            color=COL_REPLICATE_EDGE,
            fontweight='bold')
    arrow(ax, (6.75, 1.25), (7.4, 1.25),
          color=COL_PREP_EDGE,
          lw=2,
          linestyle='dashed')

    ax.set_title('PrepareModuleInput：布局转换桥',
                 fontsize=15,
                 fontweight='bold',
                 pad=6)
    save(fig, 'fsdp2_prepare_module_input.svg')


# ---------------------------------------------------------------------------
# Figure 17: Baseline TP vs Sequence Parallel
# ---------------------------------------------------------------------------
def draw_baseline_vs_sp():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2))

    def draw_flow(ax, title, steps):
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        bg = patches.FancyBboxPatch((0.06, 0.06),
                                    0.88,
                                    0.9,
                                    boxstyle='round,pad=0.02',
                                    facecolor='#fafafa',
                                    edgecolor='#bdbdbd',
                                    linewidth=1.5)
        ax.add_patch(bg)

        colors = {
            'Replicate': COL_REPLICATE,
            'Shard(1)': COL_SEQ,
            'compute': COL_COMPUTE,
            'prepare': COL_PREP,
            'output': COL_OUTPUT,
        }
        edges = {
            'Replicate': COL_REPLICATE_EDGE,
            'Shard(1)': COL_SEQ_EDGE,
            'compute': COL_COMPUTE_EDGE,
            'prepare': COL_PREP_EDGE,
            'output': COL_OUTPUT_EDGE,
        }

        n = len(steps)
        y_positions = np.linspace(0.82, 0.14, n)
        for i, (label, kind) in enumerate(steps):
            y = y_positions[i]
            rounded_box(ax, (0.14, y - 0.06),
                        0.72,
                        0.12,
                        colors[kind],
                        edges[kind],
                        text=label,
                        fontsize=10,
                        radius=0.02)
            if i < n - 1:
                arrow(ax, (0.5, y - 0.07), (0.5, y_positions[i + 1] + 0.07),
                      color='#424242',
                      lw=1.5)

    baseline_steps = [
        ('输入\n[B,S,H] Replicate', 'Replicate'),
        ('Attention / FFN\n内部分片计算', 'compute'),
        ('输出\n[B,S,H] Replicate', 'Replicate'),
    ]
    draw_flow(ax1, '基本张量并行', baseline_steps)

    sp_steps = [
        ('输入\n[B,S/tp,H] Shard(1)', 'Shard(1)'),
        ('RMSNorm\n分片计算', 'Shard(1)'),
        ('PrepareModuleInput\nShard(1) → Replicate', 'prepare'),
        ('Attention / FFN\n内部分片计算', 'compute'),
        ('输出\n[B,S/tp,H] Shard(1)', 'Shard(1)'),
    ]
    draw_flow(ax2, '序列并行', sp_steps)

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    fig.text(0.5,
             0.5,
             '进化',
             fontsize=14,
             ha='center',
             va='center',
             bbox=dict(boxstyle='round',
                       facecolor=COL_EMBED,
                       edgecolor=COL_EMBED_EDGE,
                       linewidth=2,
                       pad=0.4),
             fontweight='bold',
             color='#424242')

    plt.tight_layout()
    path = BASE_DIR / 'fsdp2_baseline_vs_sp.svg'
    fig.savefig(path, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('Saved fsdp2_baseline_vs_sp.svg')


# ---------------------------------------------------------------------------
# Figure 19: 2-D cluster topology
# ---------------------------------------------------------------------------
def draw_cluster_topology():
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 7.2)
    ax.axis('off')

    n_hosts = 8
    gpus_per_host = 8
    host_spacing = 0.78
    gpu_spacing = 0.72

    gpu_color = '#bbdefb'
    gpu_edge = '#1565c0'
    nvlink_color = '#ef6c00'
    dp_color = '#2e7d32'

    for h in range(n_hosts):
        host_y = 6.6 - h * host_spacing
        ax.text(-0.1,
                host_y,
                f'主机 {h}',
                fontsize=10,
                fontweight='bold',
                ha='right',
                va='center')

        nvlink_y = host_y - 0.2
        ax.plot([0.3, 0.3 + (gpus_per_host - 1) * gpu_spacing],
                [nvlink_y, nvlink_y],
                color=nvlink_color,
                linewidth=3,
                alpha=0.6)
        ax.text(0.3 + (gpus_per_host - 1) * gpu_spacing / 2,
                nvlink_y - 0.32,
                'TP (NVLink)',
                fontsize=8,
                ha='center',
                color=nvlink_color,
                fontweight='bold')

        for g in range(gpus_per_host):
            x = 0.3 + g * gpu_spacing
            rect = patches.FancyBboxPatch((x - 0.28, host_y - 0.22),
                                          0.56,
                                          0.44,
                                          boxstyle='round,pad=0.02',
                                          facecolor=gpu_color,
                                          edgecolor=gpu_edge,
                                          linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x,
                    host_y,
                    f'GPU\n{g}',
                    fontsize=6.5,
                    ha='center',
                    va='center')

    # DP connections
    for g in range(gpus_per_host):
        x = 0.3 + g * gpu_spacing
        ax.plot([x, x], [-0.05, 6.7],
                color=dp_color,
                linewidth=1.5,
                linestyle='--',
                alpha=0.5)

    ax.set_title('64 GPU 集群：8 主机 × 8 GPU\n主机内 NVLink 跑 TP，主机间网络跑 FSDP2',
                 fontsize=14,
                 fontweight='bold',
                 pad=8)
    save(fig, 'fsdp2_cluster_topology.svg')


# ---------------------------------------------------------------------------
# Figure 21: 2-D DeviceMesh
# ---------------------------------------------------------------------------
def draw_device_mesh_2d():
    fig, ax = plt.subplots(figsize=(7.5, 7.4))
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 7.4)
    ax.axis('off')

    mesh_size = 8
    cell_size = 0.7
    grid_color = '#bdbdbd'
    origin_x, origin_y = 0.55, 1.3

    for dp in range(mesh_size):
        for tp in range(mesh_size):
            x = origin_x + tp * cell_size
            y = origin_y + (mesh_size - 1 - dp) * cell_size
            rect = patches.Rectangle((x, y),
                                     cell_size,
                                     cell_size,
                                     facecolor='white',
                                     edgecolor=grid_color,
                                     linewidth=1)
            ax.add_patch(rect)
            ax.text(x + cell_size / 2,
                    y + cell_size / 2,
                    f'({dp},{tp})',
                    fontsize=6.5,
                    ha='center',
                    va='center',
                    color='#424242')

    # Highlight tp_mesh columns
    for tp in range(mesh_size):
        x = origin_x + tp * cell_size
        rect = patches.Rectangle((x, origin_y),
                                 cell_size,
                                 mesh_size * cell_size,
                                 facecolor='#fff3e0',
                                 edgecolor='#ef6c00',
                                 linewidth=2,
                                 alpha=0.3)
        ax.add_patch(rect)

    # Highlight dp_mesh rows
    for dp in range(mesh_size):
        y = origin_y + (mesh_size - 1 - dp) * cell_size
        rect = patches.Rectangle((origin_x, y),
                                 mesh_size * cell_size,
                                 cell_size,
                                 facecolor='#e8f5e9',
                                 edgecolor='#2e7d32',
                                 linewidth=2,
                                 alpha=0.2)
        ax.add_patch(rect)

    for tp in range(mesh_size):
        ax.text(origin_x + tp * cell_size + cell_size / 2,
                origin_y + mesh_size * cell_size + 0.12,
                f'tp={tp}',
                fontsize=7,
                ha='center',
                va='bottom',
                color=COL_REPLICATE_EDGE,
                fontweight='bold')

    for dp in range(mesh_size):
        ax.text(origin_x - 0.12,
                origin_y + (mesh_size - 1 - dp) * cell_size + cell_size / 2,
                f'dp={dp}',
                fontsize=7,
                ha='right',
                va='center',
                color=COL_REPLICATE_EDGE,
                fontweight='bold')

    # Compact legend below the grid
    legend_x = origin_x
    legend_y = origin_y - 0.55
    ax.add_patch(
        patches.Rectangle((legend_x, legend_y),
                          0.3,
                          0.3,
                          facecolor='#fff3e0',
                          edgecolor='#ef6c00',
                          linewidth=2,
                          alpha=0.5))
    ax.text(legend_x + 0.4,
            legend_y + 0.15,
            'tp_mesh: 纵向列 (主机内 TP)',
            fontsize=9,
            va='center')

    ax.add_patch(
        patches.Rectangle((legend_x + 3.0, legend_y),
                          0.3,
                          0.3,
                          facecolor='#e8f5e9',
                          edgecolor='#2e7d32',
                          linewidth=2,
                          alpha=0.5))
    ax.text(legend_x + 3.4,
            legend_y + 0.15,
            'dp_mesh: 横向行 (跨主机 FSDP2)',
            fontsize=9,
            va='center')

    ax.set_title(
        "mesh_2d = init_device_mesh('cuda', (8, 8))\n"
        '2-D DeviceMesh: [dp_size=8, tp_size=8]',
        fontsize=13,
        fontweight='bold',
        pad=8)
    save(fig, 'fsdp2_device_mesh_2d.svg')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    draw_overview()
    draw_device_mesh_1d()
    draw_transformer_block_plan()
    draw_embedding_rowwise()
    draw_attention_qkv_colwise()
    draw_attention_wo_rowwise()
    draw_mlp_w1w3_colwise()
    draw_mlp_w2_rowwise()
    draw_sequence_parallel()
    draw_sp_attention_bridge()
    draw_transformer_block()
    draw_output_colwise()
    draw_output_all_gather()
    draw_loss_parallel()
    draw_prepare_module_input()
    draw_baseline_vs_sp()
    draw_cluster_topology()
    draw_device_mesh_2d()

    print('All diagrams generated.')


if __name__ == '__main__':
    main()
