#!/usr/bin/env python
"""Generate legal old-default E centers without building or training a ternary model."""
import argparse
import fcntl
import json
import inspect
import shutil
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import torch
import tricoopdiff.diffdock_compat
from train_entry import TernaryDataLoader,_load_score_model_args,case_seed
import run_coupled_small_sample as coupled
from run_coupled_small_sample import generate_center,center_hash,validate_center_geometry
from tricoopdiff.initialization.candidates import _proposal_geometry
from tricoopdiff.models.expert import (build_diffdock_expert,configure_expert_policy,
    expert_runtime_identity)
from tricoopdiff.runtime import atomic_torch_save,sha256_file,evaluation_context
from featurize_common import esm_encoder_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('/root/.local/share/coopdiff/legacy_e_reproduction/centers'))
    parser.add_argument('--train-data',type=Path,default=ROOT/'results/small_sample_seed73/train.json')
    parser.add_argument('--unseen-data',type=Path,default=ROOT/'results/small_sample_seed73/unseen_complex_val.json')
    parser.add_argument('--include-transfer',action='store_true',help='Add 8 independent and 8 cross-case centers; total cap 32')
    parser.add_argument('--expert-model-dir',default=str(ROOT/'checkpoints/score_model'))
    parser.add_argument('--cache-dir',default=str(ROOT/'results/featurize_cache'))
    parser.add_argument('--resume-center-bank',type=Path)
    parser.add_argument('--reuse-known-1alz',type=Path)
    parser.add_argument('--correct-clock',action='store_true',help='Consume all steps+1 schedule boundaries; explicitly differs from inherited clock')
    parser.add_argument('--binary-steps',type=int,choices=(5,20),default=5)
    parser.add_argument('--max-pools-per-center',type=int,default=32)
    parser.add_argument('--memory-fraction',type=float,default=.15)
    args=parser.parse_args()
    if not 1 <= args.max_pools_per_center <= 32:
        parser.error('max-pools-per-center must be between 1 and 32')
    inherited_source=inspect.getsource(coupled.DualExpertAnchorInitializer._dock_rigid)
    needs_wrapper='actual_steps = self.binary_steps - 1' in inherited_source
    if args.correct_clock and needs_wrapper:
        class CorrectedClockInitializer(coupled.DualExpertAnchorInitializer):
            def _dock_rigid(self, *values, **keywords):
                steps = self.binary_steps
                getter = self.process.get_inference_schedule
                self.binary_steps = steps + 1
                self.process.get_inference_schedule = lambda unused: getter(steps)
                try:
                    return super()._dock_rigid(*values, **keywords)
                finally:
                    self.binary_steps = steps
                    self.process.get_inference_schedule = getter
        coupled.DualExpertAnchorInitializer = CorrectedClockInitializer
    os.chdir(ROOT)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'writer.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'manifest.json').exists():raise RuntimeError('fresh center generation requires a new output directory')
    (output/'pid').write_text(str(os.getpid()))
    device=torch.device('cuda',0)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    score_args=_load_score_model_args(args.expert_model_dir)
    encoder_identity=esm_encoder_identity()
    print(json.dumps({'event':'esm_identity_pinned','scope':'startup_full_content_read_only'}),flush=True)
    train=TernaryDataLoader(str(args.train_data),score_args,shuffle=False,device=str(device),cache_dir=args.cache_dir,cache_read_only=True,cache_wait_seconds=0,expected_esm_encoder_identity=encoder_identity)
    if len(train.entries)!=8:raise ValueError('expected all 8 training cases')
    expert=build_diffdock_expert(args.expert_model_dir,device=str(device))
    configure_expert_policy(expert,100.,'irrep_norm')
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    shutil.copyfile(__file__,snapshot/'prepare_coupled_centers.py')
    shutil.copyfile(ROOT/'tricoopdiff/initialization/candidates.py',snapshot/'candidates.py')
    protocol={'source':'old-code-default-center reconstruction, not exact original launcher',
        'binary_steps':args.binary_steps,'proposal_attempts':4,'candidate_budget':16,
        'binary_retry_limit':2,'binary_consecutive_failure_limit':3,
        'seed_policy':'first pools direct73/74; retry sequential base+attempt; original center seeds unknown; reused documented legal seeds explicit',
        'max_pools_per_center':args.max_pools_per_center,
        'strict_old_default':args.binary_steps==5 and not args.correct_clock,
        'corrected_clock':args.correct_clock,
        'clock_contract':'all steps+1 boundaries consumed' if args.correct_clock else 'inherited steps-1 with merged final dt',
        'allow_partial':False,'selection':'acceptable geometry penalty minimum',
        'native_information_used':False,'sdpi_fallback':False,
        'effective_sampler':{'mode':'local_wrapper' if args.correct_clock and needs_wrapper else 'formal_shared_sampler',
            'source_sha256':sha256_file(snapshot/'candidates.py'),
            'entry_source_sha256':sha256_file(snapshot/'prepare_coupled_centers.py'),
            'snapshot':str(snapshot)},
        'expert_runtime':expert_runtime_identity(expert),'script_sha256':sha256_file(__file__),
        'center_implementation_sha256':sha256_file(ROOT/'scripts/run_coupled_small_sample.py'),
        'train_index_sha256':sha256_file(args.train_data),'pid':os.getpid(),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'include_transfer':args.include_transfer}
    centers={'train':{},'independent_center':{},'unseen_complex':{}}
    imports=[]
    def import_record(case,record,path):
        center=record['center']
        cache=torch.load(Path(args.cache_dir)/(case+'.pt'),map_location='cpu',weights_only=False)
        geometry,valid,_=_proposal_geometry(cache['e3_data']['coords'],center['poi'].reshape(-1,3),center['ligand'].reshape(-1,3),separation=center.get('geometry',{}).get('poi_outward_separation',0.))
        validate_center_geometry({**center,'geometry':geometry,'acceptable':valid and center.get('acceptable') is True},case=case,seed=record['seed'])
        if center_hash(center)!=record['sha256']:raise ValueError('imported center hash mismatch')
        if any(record['sha256']==old['sha256'] for old in centers['train'].get(case,[])):raise ValueError('duplicate imported center')
        record={**record,'imported_from':str(path),'imported_bank_sha256':sha256_file(path)}
        centers['train'].setdefault(case,[]).append(record)
        imports.append({'case':case,'seed':record['seed'],'center_sha256':record['sha256'],'path':str(path),'source_file_sha256':sha256_file(path)})
    if args.resume_center_bank:
        previous=torch.load(args.resume_center_bank,map_location='cpu',weights_only=False)
        for case,records in previous['train'].items():
            for record in records:import_record(case,record,args.resume_center_bank)
        for group in ('independent_center','unseen_complex'):
            for case,record in previous.get(group,{}).items():
                cache=torch.load(Path(args.cache_dir)/(case+'.pt'),map_location='cpu',weights_only=False)
                center=record['center']
                geometry,valid,_=_proposal_geometry(cache['e3_data']['coords'],center['poi'].reshape(-1,3),center['ligand'].reshape(-1,3),separation=center.get('geometry',{}).get('poi_outward_separation',0.))
                validate_center_geometry({**center,'geometry':geometry,'acceptable':valid and center.get('acceptable') is True},case=case,seed=record['seed'])
                if record.get('native_information_used') is not False or center_hash(center)!=record['sha256']:
                    raise ValueError(f'{case}: invalid resumed transfer center identity')
                centers[group][case]=record
    if args.reuse_known_1alz:
        known=torch.load(args.reuse_known_1alz,map_location='cpu',weights_only=False)
        if not centers['train'].get('1ALZ_A_B_EOH'):import_record('1ALZ_A_B_EOH',known,args.reuse_known_1alz)
    protocol['imported_legal_centers']=imports
    done=sum(len(records) for records in centers['train'].values())+sum(len(centers[group]) for group in ('independent_center','unseen_complex'));started=time.monotonic()
    atomic_torch_save(centers,output/'predicted_centers.partial.pt')
    (output/'start_manifest.json').write_text(json.dumps(protocol,indent=2))
    def save_center(batch,group,index,seed):
        nonlocal done
        case=batch['case'];before=time.monotonic()
        attempts=[]
        attempt_dir=output/'attempts';attempt_dir.mkdir(exist_ok=True)
        record=None
        for attempt in range(args.max_pools_per_center):
            actual_seed=(seed+attempt) % (2**31-1)
            (output/'progress.json').write_text(json.dumps({'status':'generating',
                'completed_centers':done,'case':case,'group':group,'center_index':index,
                'pool_attempt':attempt,'seed':actual_seed,'elapsed_seconds':time.monotonic()-started},indent=2))
            if actual_seed in {old['seed'] for old in centers['train'].get(case,[])}:
                print(json.dumps({'event':'skip_existing_seed','case':case,'seed':actual_seed}),flush=True)
                continue
            attempt_started=time.monotonic()
            try:
                record=generate_center(batch,expert,actual_seed,
                    binary_steps=args.binary_steps,proposal_attempts=4)
                summary={'case':case,'group':group,'center_index':index,'attempt':attempt,
                    'seed':actual_seed,'status':'accepted','elapsed_seconds':time.monotonic()-attempt_started,
                    'selection_diagnostics':record['selection_diagnostics']}
            except (ValueError,FloatingPointError) as error:
                detail=str(error)
                diagnostic=json.loads(detail.split('diagnostics=',1)[1]) if 'diagnostics=' in detail else None
                summary={'case':case,'group':group,'center_index':index,'attempt':attempt,
                    'seed':actual_seed,'status':'rejected','elapsed_seconds':time.monotonic()-attempt_started,
                    'error':detail,'selection_diagnostics':diagnostic}
                record=None
            if record is not None and any(record['sha256']==old['sha256'] for old in centers['train'].get(case,[])):
                summary['status']='rejected_duplicate_center'
                record=None
            attempts.append(summary)
            path=attempt_dir/f'{case}.{group}.{index}.pool{attempt}.json'
            path.write_text(json.dumps(summary,indent=2))
            print(json.dumps({'event':'center_pool_attempt','case':case,'group':group,
                'center_index':index,'attempt':attempt,'seed':actual_seed,'status':summary['status'],
                'elapsed_seconds':summary['elapsed_seconds']}),flush=True)
            if record is not None:break
        if record is None:
            raise ValueError(f'{case}: exhausted {args.max_pools_per_center} deterministic native-free pools for {group} center {index}; no case dropped or SDPI fallback')
        record['pool_attempts']=len(attempts)
        record['base_seed']=seed
        record['generation_elapsed_seconds']=time.monotonic()-before
        if group=='train':centers['train'].setdefault(case,[]).append(record)
        else:centers[group][case]=record
        done+=1
        atomic_torch_save(centers,output/'predicted_centers.partial.pt')
        (output/'progress.json').write_text(json.dumps({'status':'generating','completed_centers':done,'case':case,'elapsed_seconds':time.monotonic()-started},indent=2))
        print(json.dumps({'event':'center_persisted','case':case,'group':group,'center_index':index,'completed_centers':done,'generation_elapsed_seconds':record['generation_elapsed_seconds'],'acceptable_count':record['selection_diagnostics']['acceptable_count']}),flush=True)
    try:
        with evaluation_context(expert):
            for batch in train:
                case=batch['case']
                for index in range(2):
                    if index < len(centers['train'].get(case,[])):continue
                    save_center(batch,'train',index,73+index)
                if args.include_transfer and case not in centers['independent_center']:save_center(batch,'independent_center',0,case_seed(f'{case}:independent_center',7400))
            if args.include_transfer:
                unseen=TernaryDataLoader(str(args.unseen_data),score_args,shuffle=False,device=str(device),cache_dir=args.cache_dir,cache_read_only=True,cache_wait_seconds=0,expected_esm_encoder_identity=encoder_identity)
                if len(unseen.entries)!=8:raise ValueError('expected all 8 cross-case cases')
                for batch in unseen:
                    if batch['case'] not in centers['unseen_complex']:
                        save_center(batch,'unseen_complex',0,case_seed(f"{batch['case']}:unseen_center",7600))
        expected=32 if args.include_transfer else 16
        if done!=expected:raise RuntimeError('center coverage mismatch')
        atomic_torch_save(centers,output/'predicted_centers.pt')
        protocol.update(status='complete',completed_centers=done,elapsed_seconds=time.monotonic()-started,center_bank_sha256=sha256_file(output/'predicted_centers.pt'))
        (output/'manifest.json').write_text(json.dumps(protocol,indent=2))
        print(json.dumps({'event':'complete',**protocol}),flush=True)
    except Exception as error:
        failure={**protocol,'status':'failed','completed_centers':done,'elapsed_seconds':time.monotonic()-started,'error':str(error),'error_type':type(error).__name__}
        (output/'failure.json').write_text(json.dumps(failure,indent=2))
        print(json.dumps({'event':'failed',**failure}),flush=True)
        raise


if __name__=='__main__':main()
