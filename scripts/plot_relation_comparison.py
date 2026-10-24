"""Plot aggregate final-test relation results from an ignored experiment report."""
import argparse
import json
from pathlib import Path

APPROACHES = ('rules', 'original', 'expanded', 'abstention')
LABELS = ('Conservative rules', 'Learned links · 81 training notes',
          'Expanded training · 95 notes', 'Training-fold abstention')


def aggregates(report):
    if set(report['results']) != set(APPROACHES) or type(report['notes']) is not int or report['notes'] < 1:
        raise ValueError('Expected four approaches and a positive note count')
    values = []
    for name in APPROACHES:
        result = report['results'][name]
        f1 = result['complete_records']['f1']
        false_links = sum(result[f]['fp'] for f in ('dose', 'route', 'frequency'))
        if f1 is None or not 0 <= f1 <= 1 or type(false_links) is not int or false_links < 0:
            raise ValueError('Invalid aggregate performance')
        values.append((100*f1, false_links))
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('docs/images/relation-test-comparison.svg'))
    args=parser.parse_args()
    report=json.loads(args.report.read_text());values=aggregates(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'final-relation-comparison'})
    fig,(left,right)=plt.subplots(1,2,figsize=(10.4,4.8),gridspec_kw={'width_ratios':[1.5,1]})
    fig.subplots_adjust(left=.32,right=.95,top=.72,bottom=.24,wspace=.32)
    fig.suptitle('Medication linking: final held-out comparison',x=.04,y=.96,ha='left',fontsize=16,weight='bold',color='#193344')
    fig.text(.04,.85,f'{report["notes"]} reserved test notes · identical predicted entities · frozen models and thresholds',color='#536d78')
    colors=['#135e63','#5378a1','#9b6a39','#8b699d']
    for axis,column in ((left,0),(right,1)):
        numbers=[v[column] for v in values]
        axis.barh(range(4),numbers,color=colors,height=.56)
        axis.invert_yaxis();axis.set_yticks(range(4),LABELS if column==0 else ['']*4)
        axis.tick_params(axis='y',length=0,pad=10)
        axis.set_axisbelow(True);axis.grid(axis='x',alpha=.18)
        for spine in ('top','right','left'):axis.spines[spine].set_visible(False)
        for i,number in enumerate(numbers):
            axis.text(number+(1.5 if column==0 else .15),i,f'{number:.1f}%' if column==0 else str(number),va='center',fontsize=10)
    left.set_xlim(0,105);left.set_xticks([0,25,50,75,100]);left.xaxis.set_major_formatter(PercentFormatter())
    left.set_title('Complete-record F1 ↑',loc='left',pad=14)
    right.set_xlim(0,max(2,max(v[1] for v in values)*1.25));right.xaxis.set_major_locator(MaxNLocator(integer=True))
    right.set_title('False medication–attribute links ↓',loc='left',pad=14,fontsize=10)
    fig.text(.04,.09,'Source-checked teacher agreement, not clinical accuracy. Dose/route/frequency references are candidate-limited.\n'
             'Rules were selected on development data; the test comparison does not select or retune the demo.',fontsize=9,color='#536d78')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,metadata={'Date':None} if args.output.suffix=='.svg' else None)
    plt.close(fig)


if __name__=='__main__':main()
