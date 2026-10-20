"""CPU tests of functions extracted unchanged from the supplied training source.

Extracting AST functions isolates numerical tests from CUDA/Megatron imports.
Only the ActorConfig type and registry decorator are lightweight test scaffolding.
This is not a distributed-training integration test.
"""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'vendor/verl/verl/trainer/ppo/core_algos.py'

class ActorConfig:
    policy_loss={'prefix_exact_kl_delta_low':.1,'prefix_exact_kl_delta_high':.1}
    global_batch_info={}
    def __init__(self,values=None):
        if values: self.__dict__.update(values)
    def get(self,key,default=None):return getattr(self,key,default)

def load_functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    prefix=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[prefix,*nodes],type_ignores=[]))
    exec(compile(module,str(path),'exec'),namespace)

NS={'torch':torch,'np':np,'os':os,'ActorConfig':ActorConfig,'AlgoConfig':type('AlgoConfig',(),{}),'register_policy_loss':lambda name:lambda f:f}
FUNCS=['masked_sum','masked_mean']
load_functions(ROOT/'vendor/verl/verl/utils/torch_functional.py',FUNCS,NS)
NS['verl_F']=SimpleNamespace(**{k:NS[k] for k in FUNCS})
MODES=['prefix_geometric_probability_weighted_exact_kl_dual_clip',
       'prefix_probability_weighted_exact_kl_dual_clip',
       'prefix_geometric_probability_weighted_exact_kl_cumulative_dual_clip']
BASELINES=['compute_policy_loss_vanilla','compute_policy_loss_gspo','compute_policy_loss_ripo_clip','compute_policy_loss_cum_token_cumprod_la']
load_functions(SOURCE,['agg_loss','_solve_exact_prefix_kl_log_bounds','_compute_policy_loss_prefix_exact_kl_clip',
    '_dump_prefix_clip_response_diagnostics',*['compute_policy_loss_'+m for m in MODES],*BASELINES],NS)
os.environ['VERL_PREFIX_CLIP_DIAG_DISABLE']='1'

class KernelTests(unittest.TestCase):
    def test_all_baseline_kernels(self):
        for method,name in zip(('dapo','gspo','ripo','ctpo'),BASELINES):
            cfg=ActorConfig(json.loads((ROOT/'configs'/f'{method}.json').read_text())['actor_rollout_ref']['actor'])
            old=torch.full((2,8),-2.)
            current=(old+torch.linspace(-.02,.02,8)).requires_grad_()
            mask=torch.ones_like(old);mask[0,-2:]=0
            adv=torch.ones_like(old);adv[1]=-1
            loss,metrics=NS[name](old,current,adv,mask,loss_agg_mode=cfg.loss_agg_mode,config=cfg)
            loss.backward()
            self.assertTrue(torch.isfinite(loss),name)
            self.assertTrue(torch.isfinite(current.grad).all(),name)
            self.assertIn('actor/pg_clipfrac',metrics)

    def test_exact_coordinate_roots(self):
        budgets=torch.logspace(-8,4,80,dtype=torch.float64)
        lo,hi=NS['_solve_exact_prefix_kl_log_bounds'](budgets.log())
        for z in (lo,hi):
            self.assertTrue(torch.allclose(torch.expm1(z)-z,budgets,rtol=2e-4,atol=1e-10))
        self.assertTrue(torch.all(lo<0) and torch.all(hi>0))
        self.assertFalse(lo.requires_grad)
        l,h=NS['_solve_exact_prefix_kl_log_bounds'](torch.tensor([1000.]))
        self.assertTrue(torch.isneginf(l).all() and torch.isfinite(h).all())

    def test_current_token_gradient_and_ratio_variants(self):
        for mode in MODES:
            old=torch.full((1,3),-2.,dtype=torch.float64)
            current=(old+.02).requires_grad_()
            adv=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
            loss,_=NS['compute_policy_loss_'+mode](old,current,adv,torch.ones_like(old),config=ActorConfig())
            loss.backward()
            self.assertEqual(current.grad[0,0].item(),0)
            self.assertEqual(current.grad[0,1].item(),0)
            ratio=np.exp(.06 if 'cumulative' in mode else .02)
            self.assertAlmostEqual(loss.item(),-ratio/(3+1e-8),places=7)
            self.assertAlmostEqual(current.grad[0,2].item(),-ratio/(3+1e-8),places=7)

    def test_dual_clip_and_masking(self):
        old=torch.full((2,4),-2.)
        current=(old+2.).requires_grad_()
        mask=torch.tensor([[1.,1.,1.,0.],[1.,1.,0.,0.]])
        adv=torch.tensor([[1.,1.,1.,0.],[-1.,-1.,0.,0.]])
        loss,metrics=NS['compute_policy_loss_'+MODES[0]](old,current,adv,mask,config=ActorConfig())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(current.grad,torch.zeros_like(current.grad)))
        self.assertGreater(metrics['actor/'+MODES[0]+'/dual_clipfrac'],0)

if __name__=='__main__':unittest.main()
