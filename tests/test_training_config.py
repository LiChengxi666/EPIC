import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_experiment import available_methods,materialize

class TrainingConfigTests(unittest.TestCase):
    def test_configs_and_registered_losses(self):
        self.assertEqual(set(available_methods()),{'epic','grpo','dapo','gspo','ctpo','ripo'})
        source=ast.parse((ROOT/'vendor/verl/verl/trainer/ppo/core_algos.py').read_text())
        registered=set()
        for node in source.body:
            for d in getattr(node,'decorator_list',[]):
                if isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=='register_policy_loss':
                    registered.add(ast.literal_eval(d.args[0]))
        for method in available_methods():
            c=materialize(method,'/model','/data','/output')
            self.assertEqual(c['trainer']['experiment_name'],method)
            self.assertNotIn('__MODEL_PATH__',json.dumps(c))
            self.assertNotIn('__OUTPUT_DIR__',json.dumps(c))
            self.assertIn(c['actor_rollout_ref']['actor']['policy_loss']['loss_mode'],registered)
            self.assertEqual(c['trainer']['nnodes']*c['trainer']['n_gpus_per_node'],32)
            self.assertEqual(c['actor_rollout_ref']['actor']['megatron']['router_replay']['mode'],'disabled')

    def test_readme_config(self):
        text=(ROOT/'README.md').read_text()
        config=materialize('epic','/model','/data','/output')['actor_rollout_ref']['actor']
        self.assertIn(config['policy_loss']['loss_mode'],text)
        self.assertIn(config['loss_agg_mode'],text)
        self.assertNotIn('experiments.json',text)
        self.assertNotIn('![',text)

    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            run=subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),'epic',
                '--model',tmp,'--data',tmp,'--output',str(out)],capture_output=True,text=True,check=True)
            config=json.loads(run.stdout)
            self.assertEqual(config['trainer']['default_local_dir'],str((out/'checkpoints').resolve()))
            self.assertFalse(out.exists())

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            materialize('../anything','/model','/data','/output')

    def test_no_paper_bundle(self):
        for name in ('experiments.json','data_manifest.json','results','assets'):
            self.assertFalse((ROOT/name).exists())
        self.assertFalse(list((ROOT/'configs').glob('exp*.json')))

if __name__=='__main__': unittest.main()
