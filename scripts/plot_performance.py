"""Render aggregate paired-seed performance for the README (requires matplotlib)."""
import argparse
from pathlib import Path

# Exact name-span F1 on the same ten frozen, AI-reviewed development notes.
PAIRED_F1 = {
    42: (0.8932038834951457, 0.912621359223301),
    7: (0.8712871287128713, 0.9215686274509803),
    123: (0.9702970297029703, 0.9387755102040817),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/images/training-performance.svg'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'medication-performance'})
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    fig.subplots_adjust(left=.12, right=.77, top=.77, bottom=.21)
    fig.suptitle('Medication-name extraction: training expansion', x=.12, ha='left',
                 fontsize=16, weight='bold', color='#193344')
    means = [100*sum(pair[i] for pair in PAIRED_F1.values())/len(PAIRED_F1) for i in range(2)]
    fig.text(.12, .865, f'Mean F1: {means[0]:.1f}% → {means[1]:.1f}%  (+{means[1]-means[0]:.1f} percentage points)',
             color='#135e63', fontsize=12)
    for (seed, pair), color in zip(PAIRED_F1.items(), ['#5378a1', '#9b6a39', '#8b699d']):
        values = [100*value for value in pair]
        ax.plot([0, 1], values, marker='o', color=color, linewidth=1.8, markersize=6)
        offset = -12 if seed == 42 else (0 if seed == 7 else 8)
        ax.annotate(f'Seed {seed}: {values[1]:.1f}%', (1, values[1]), xytext=(14, offset),
                    textcoords='offset points', color=color, va='center', fontsize=10)
    ax.plot([0, 1], means, color='#135e63', linewidth=3, marker='D', markersize=7)
    ax.annotate('Mean', (.56, means[0]+.56*(means[1]-means[0])), xytext=(0, 13),
                textcoords='offset points', color='#135e63', weight='bold')
    ax.set_xlim(-.12, 1.12)
    ax.set_ylim(80, 100)
    ax.set_yticks([80, 85, 90, 95, 100])
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel('Teacher-reference exact name-span F1')
    ax.set_xticks([0, 1], ['44 training notes\n65 optimizer steps', '81 training notes\n115 optimizer steps'])
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    fig.text(.12, .065, '10 fixed development notes · 3 paired seeds · 5 epochs\n'
                       'Both pools include 8 additional synthetic negatives. Teacher agreement is not clinical accuracy.',
             fontsize=9, color='#536d78')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={'Date': None} if args.output.suffix == '.svg' else None)
    plt.close(fig)


if __name__ == '__main__':
    main()
