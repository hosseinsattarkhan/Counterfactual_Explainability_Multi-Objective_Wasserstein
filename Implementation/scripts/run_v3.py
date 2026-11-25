import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from evaluation.benchmark_v3 import run
out=Path(__file__).resolve().parents[2]/'Experiments/results/v3_benchmark.json'
r=run(out)
from collections import Counter
print('Runs:',len(r['runs']),'Statuses:',dict(Counter(x['status'] for x in r['runs'])))
print('Saved:',out)
