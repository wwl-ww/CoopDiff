#!/usr/bin/env python
"""Compare a rollout-augmented pilot with its factorial base checkpoint."""

import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).parent)]
import tricoopdiff.diffdock_compat
from factorial_arm import PairedTorsionFlow,restrict_non_single
from train_entry import TernaryDataLoader,_load_score_model_args
from tricoopdiff.consensus import ConsensusFuse
from tricoopdiff.consensus_experiment import ExperimentalConsensus,ExperimentalConsensusTrainer,RolloutAugmentedTrainer
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.sampler import sample_single
from tricoopdiff.evaluate import compute_rmsd,compute_deepternary_dockq

def build(path):
 c=torch.load(path,map_location='cuda',weights_only=False);expert=build_diffdock_expert('/tmp/coopdiff/checkpoints/score_model','cuda');fuse=ConsensusFuse(**c['model_config']);model=ExperimentalConsensus(expert,fuse,c['expert_lora_config']);r=c['runtime_config'];common=dict(device='cuda',experiment_contract=c['experiment_contract'],lr=5e-5 if 'rollout_probability' in r else 1e-4,lora_lr=1e-5,lora_early_lr=5e-6,lora_warmup_steps=0,activation_clamp=r['activation_clamp'],activation_clamp_mode=r['activation_clamp_mode'],residual_weight=r['residual_weight'],ema_decay=r['ema_decay']);trainer=(RolloutAugmentedTrainer(model,PairedTorsionFlow(),rollout_probability=r['rollout_probability'],rollout_steps=r['rollout_steps'],**common) if 'rollout_probability' in r else ExperimentalConsensusTrainer(model,PairedTorsionFlow(),**common));trainer.load_checkpoint(path);return model,trainer

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();idx=Path('/tmp/coopdiff/results/overnight_root_cause_v3_plan/indexes');score=_load_score_model_args('/tmp/coopdiff/checkpoints/score_model');loader=TernaryDataLoader(idx/'ternary_val_ccd.json',score,shuffle=False,device='cuda',cache_dir='/tmp/coopdiff/results/overnight_root_cause_v3_plan/cache_ccd');entries=json.loads((idx/'ternary_val_ccd.json').read_text())['entries'];paths={e['case']:e['ligand'] for e in entries};wanted={'3CXH_C_E_6PH','6U9D_Q_R_TPP','4I0P_D_H_PGE','5W7C_A_C_FTT','6HS4_A_B_GON','4IL6_C_I_LMT','7L0F_G_H_GSP','3ZCB_A_B_ATP'};batches=[restrict_non_single(b,paths[b['case']]) for b in loader if b['case'] in wanted];rows=[]
 for label,path in [('base',args.base),('rollout',args.pilot)]:
  model,trainer=build(path)
  with trainer.ema_parameters(),torch.no_grad():
   model.eval()
   for b in batches:
    native_lig=b['ligand_data']['coords'][0].cpu().numpy();native_poi=b['poi_data']['coords'][0].cpu().numpy();e3=b['e3_data']['coords'][0].cpu().numpy()
    for seed in [12001,12002]:
     result=sample_single(model,trainer.diffusion,TernaryInitializer(),b['e3_data'],b['poi_data'],b['ligand_data'],steps=20,device='cuda',seed=seed);pred=result['ligand_coords'];poi=result['poi_coords'];dock=compute_deepternary_dockq(native_lig,e3,native_poi,pred,e3,poi);rows.append({'model':label,'case':b['case'],'seed':seed,'valid':result['valid'],'protac_rmsd':compute_rmsd(pred,native_lig,align_first=True),'ligand_center_error':float(np.linalg.norm(pred.mean(0)-native_lig.mean(0))),'poi_center_error':float(np.linalg.norm(poi.mean(0)-native_poi.mean(0))),'approx_dockq':dock['dockq'] if dock else None})
    print(label,b['case'],len(rows),flush=True)
  del trainer,model;torch.cuda.empty_cache()
 summary=[]
 for label in ['base','rollout']:
  x=[r for r in rows if r['model']==label];summary.append({'model':label,**{k:float(np.mean([r[k] for r in x])) for k in ['protac_rmsd','ligand_center_error','poi_center_error','approx_dockq']}})
 args.out.write_text(json.dumps({'scope':'first 8 CCD validation cases x 2 paired seeds; 20-step free rollout','summary':summary,'rows':rows},indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
