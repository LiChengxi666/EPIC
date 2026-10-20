"""Portable launcher. Default is a dry run; --execute is required for training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]

def available_methods():
    return sorted(p.stem for p in (ROOT/'configs').glob('*.json'))

def materialize(method,model,data,output,ray_address='auto'):
    if method not in available_methods(): raise ValueError('Unknown method: '+method)
    template=json.loads((ROOT/'configs'/f'{method}.json').read_text())
    replacements={'__MODEL_PATH__':str(model),'__DATA_DIR__':str(data),'__OUTPUT_DIR__':str(output)}
    def replace(v):
        if isinstance(v,dict): return {k:replace(x) for k,x in v.items()}
        if isinstance(v,list): return [replace(x) for x in v]
        if isinstance(v,str):
            for before,after in replacements.items(): v=v.replace(before,after)
        return v
    config=replace(template)
    config['ray_kwargs']['ray_init']['address']=ray_address
    return config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method',nargs='?',choices=available_methods())
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--model',type=Path,default=Path('model/Qwen3-30B-A3B-Base'))
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--ray-address',default='auto')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.list:
        print('\n'.join(available_methods()))
        return
    if not args.method: parser.error('Specify a method or --list')
    output=(args.output or Path('outputs')/args.method).resolve()
    config=materialize(args.method,args.model.resolve(),args.data.resolve(),output,args.ray_address)
    if not args.execute:
        print(json.dumps(config,indent=2));return
    paths=[args.model,args.data/'math-17k.parquet',*[args.data/'evals'/f'{n}.parquet' for n in ('amc23','aime24','aime25','hmmt25')]]
    for path in paths:
        if not path.exists(): parser.error(f'Required local input missing: {path}')
    if output.exists() and any(output.iterdir()): parser.error('Use a new empty output directory to avoid overwriting an experiment')
    output.mkdir(parents=True,exist_ok=True)
    vendor=ROOT/'vendor/verl'
    sys.path.insert(0,str(vendor))
    os.environ['WANDB_MODE']='disabled'
    os.environ['VERL_FILE_LOGGER_PATH']=str(output/'metrics.jsonl')
    config['ray_kwargs']['ray_init']['runtime_env']={
        'working_dir':str(vendor),
        'env_vars':{'WANDB_MODE':'disabled','VERL_FILE_LOGGER_PATH':str(output/'metrics.jsonl')}}
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2)+'\n')
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    run_ppo(OmegaConf.create(config))

if __name__=='__main__': main()
