"""Validate processed training/evaluation schema without printing examples."""
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'math-17k.parquet':(17917,'lighteval/MATH'),
    'evals/amc23.parquet':(40,'amc23'),'evals/aime24.parquet':(30,'aime24'),
    'evals/aime25.parquet':(30,'aime25'),'evals/hmmt25.parquet':(30,'hmmt25')}

def inspect_data(directory,check_counts=False):
    reports=[]
    for relative,(count,source) in EXPECTED.items():
        rows=pq.read_table(directory/relative).to_pylist()
        if not rows: raise ValueError(f'{relative}: empty input')
        if check_counts and len(rows)!=count: raise ValueError(f'{relative}: expected {count} rows, got {len(rows)}')
        for r in rows:
            if r['data_source']!=source or not isinstance(r['prompt'],list):
                raise ValueError(f'{relative}: incorrect source or prompt schema')
            if not all(set(('role','content'))<=m.keys() for m in r['prompt']):
                raise ValueError(f'{relative}: malformed message')
            if 'ground_truth' not in r['reward_model']:raise ValueError(f'{relative}: missing answer')
            if r.get('ability')!='MATH' or 'style' not in r['reward_model']:
                raise ValueError(f'{relative}: missing ability or reward style')
        if source=='amc23' and len({json.dumps(r['prompt'],sort_keys=True) for r in rows})!=len(rows):
            raise ValueError('AMC23 contains duplicate prompts')
        reports.append({'file':relative,'rows':len(rows),'data_source':source})
    return reports

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--check-counts',action='store_true',help='Also check the default dataset sizes')
    args=p.parse_args()
    report=inspect_data(args.directory,args.check_counts)
    for entry in report: print(entry['file'],entry['rows'],'rows: schema OK')

if __name__=='__main__': main()
