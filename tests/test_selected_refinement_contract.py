"""CPU-only regression tests for refinement provenance and oracle union accounting."""

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import evaluate_selected_refinement as evaluation
from tricoopdiff.condition_manifest import MANIFEST_VERSION, canonical_isomeric_smiles


def _metrics(dockq):
    return {'dockq': dockq, 'fnat': 0.3, 'lrms': 2.0, 'irms': 1.0}


def test_union_keeps_initialization_when_refinement_fails():
    row = evaluation.refinement_metric_fields(_metrics(0.6))
    assert row['safe_dockq'] == 0.6 and row['safe_stage'] == 'initial'
    assert 'refined_dockq' not in row, 'a failed refinement is not a valid final pose'
    assert evaluation.summarize([row], 'safe_dockq')['hits_0.49'] == 1
    assert evaluation.summarize([row], 'refined_dockq')['count'] == 0


def test_runtime_cache_defaults_to_asset_repository_without_overriding_explicit_value():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.dict(os.environ, {}, clear=True):
            evaluation.configure_runtime_cache(root)
            assert os.environ['TRICOOPDIFF_CACHE_DIR'] == str(
                root / 'results/runtime_cache/diffdock')
        with patch.dict(os.environ, {'TRICOOPDIFF_CACHE_DIR': '/explicit'}, clear=True):
            evaluation.configure_runtime_cache(root)
            assert os.environ['TRICOOPDIFF_CACHE_DIR'] == '/explicit'


def test_union_selects_better_stage_and_rejects_nonfinite_scores():
    for final, expected, stage in ((0.2, 0.4, 'initial'), (0.7, 0.7, 'refined')):
        row = evaluation.refinement_metric_fields(_metrics(0.4), _metrics(final))
        assert row['safe_dockq'] == expected and row['safe_stage'] == stage
    try:
        evaluation.refinement_metric_fields(_metrics(float('nan')))
    except ValueError:
        pass
    else:
        raise AssertionError('a non-finite initialization must not enter the union')


def test_resume_rejects_mismatch_legacy_and_overwrite_without_mutations():
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        manifest = {'cases': ['demo'], 'input_signatures': {'demo': 'current'}}
        report = {'case': 'demo', 'status': 'complete', 'config': manifest}
        evaluation.atomic_json(manifest, out / 'manifest.json')
        evaluation.atomic_json(report, out / 'demo.json')
        assert 'demo' in evaluation.validate_resume_output(out, manifest)
        original = {path.name: path.read_bytes() for path in out.iterdir()}
        for config, resume in (
            ({**manifest, 'native_root': 'different'}, True),
            ({**manifest, 'input_signatures': {'demo': 'changed'}}, True),
            (manifest, False),
        ):
            try:
                evaluation.validate_resume_output(out, config, resume=resume)
            except ValueError as error:
                assert '--out-dir' in str(error)
            else:
                raise AssertionError('unsafe reuse or overwrite must be rejected')
            assert {path.name: path.read_bytes() for path in out.iterdir()} == original
        evaluation.atomic_json({'cases': ['demo']}, out / 'manifest.json')
        try:
            evaluation.validate_resume_output(out, manifest)
        except ValueError:
            pass
        else:
            raise AssertionError('unsigned legacy output must not be silently adopted')


def test_condition_preflight_catches_receptor_node_mismatch():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def pdb_line(serial, residue, number, x):
        return (f'ATOM  {serial:5d}  CA  {residue:>3} A{number:4d}    '
                f'{x:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{20.:6.2f}          C \n')

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        protein = pdb_line(1, 'ALA', 1, 0.) + pdb_line(2, 'GLY', 2, 10.) + 'END\n'
        for side in (1, 2):
            (root / f'protein{side}.pdb').write_text(protein)
        molecule = Chem.AddHs(Chem.MolFromSmiles('CC'))
        AllChem.EmbedMolecule(molecule, randomSeed=7)
        ligand = root / 'ligand.sdf'
        Chem.MolToMolFile(molecule, str(ligand))
        molecule = Chem.RemoveHs(molecule)
        manifest = {
            'schema': MANIFEST_VERSION, 'case': 'fixture',
            'ligand_atom_count': molecule.GetNumAtoms(),
            'ligand_canonical_isomeric_smiles': canonical_isomeric_smiles(molecule),
            'sides': [
                {'provenance': 'template', 'evidence_id': 'left',
                 'pocket_probabilities': [1., 0.]},
                {'provenance': 'template', 'evidence_id': 'right',
                 'pocket_probabilities': [1., 0.]},
            ],
            'linker_indices': [], 'protac': True,
        }
        condition = root / 'fixture.json'
        condition.write_text(json.dumps(manifest))
        paths = {'protein1': root / 'protein1.pdb', 'protein2': root / 'protein2.pdb',
                 'ligand': ligand}
        case_inputs = {'fixture': (paths, root / 'unused_native.pdb', {})}
        audit = evaluation.preflight_condition_manifests(
            {'fixture': condition}, case_inputs)
        assert audit['fixture']['protein_sizes'] == [2, 2]
        manifest['sides'][0]['pocket_probabilities'].append(0.)
        condition.write_text(json.dumps(manifest))
        try:
            evaluation.preflight_condition_manifests({'fixture': condition}, case_inputs)
        except ValueError as error:
            assert 'expected finite shape (2,)' in str(error)
        else:
            raise AssertionError('misaligned receptor pocket passed preflight')


def test_main_checks_all_inputs_before_writing_or_loading_models():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        out = root / 'out'
        entry = {'case': 'demo'}
        for key in ('protein1', 'protein2', 'ligand', 'metric_protein1', 'metric_protein2'):
            entry[key] = key + '.pdb'
            (root / entry[key]).write_text('original structure')
        native = root / 'native' / 'demo' / 'gt_complex.pdb'
        native.parent.mkdir(parents=True)
        native.write_text('original native')
        other_native = root / 'other_native' / 'demo' / 'gt_complex.pdb'
        other_native.parent.mkdir(parents=True)
        other_native.write_text('other native')
        (root / 'index.json').write_text(json.dumps({'entries': [entry]}))
        (root / 'model.pt').write_bytes(b'not loaded in this CPU test')
        expert = root / 'checkpoints' / 'score_model'
        expert.mkdir(parents=True)
        for name in ('model_parameters.yml', 'best_ema_inference_epoch_model.pt'):
            (expert / name).write_bytes(b'original expert asset')
        scorer = root / 'DeepTernary' / 'DockQ' / 'dockq_util.py'
        scorer.parent.mkdir(parents=True)
        scorer.write_text('# original metric implementation')
        argv = ['evaluate', '--index', str(root / 'index.json'), '--out-dir', str(out),
                '--checkpoint', str(root / 'model.pt'), '--device', 'cpu',
                '--native-root', 'native']
        old_cwd = Path.cwd()
        try:
            with patch.object(evaluation, 'ROOT', root), patch.object(sys, 'argv', argv), \
                    patch.object(evaluation, 'EvaluationRunner', side_effect=RuntimeError('CPU probe')) as runner, \
                    contextlib.redirect_stdout(io.StringIO()):
                # Obtain the actual manifest without constructing any model.
                try:
                    evaluation.main()
                except SystemExit as error:
                    assert error.code == 1
                assert runner.call_count == 1
                manifest = json.loads((out / 'manifest.json').read_text())
                evaluation.atomic_json({'case': 'demo', 'config': manifest, 'status': 'complete',
                                        'arms': {}, 'refinements': []}, out / 'demo.json')
                runner.reset_mock()
                evaluation.main()
                runner.assert_not_called()
                originals = {p.name: p.read_bytes() for p in out.iterdir()}
                # Same paths but changed contents, plus a changed native-root CLI argument.
                mutations = [root / entry[key] for key in entry if key != 'case']
                mutations += [native, scorer, expert / 'model_parameters.yml',
                              expert / 'best_ema_inference_epoch_model.pt', root / 'model.pt', None]
                for path in mutations:
                    saved = path.read_bytes() if path is not None else None
                    if path is not None:
                        path.write_bytes(saved + b' changed')
                    else:
                        argv[-1] = 'other_native'
                    try:
                        evaluation.main()
                    except ValueError as error:
                        assert '--out-dir' in str(error)
                    else:
                        raise AssertionError(f'changed input was silently reused: {path}')
                    finally:
                        if path is not None:
                            path.write_bytes(saved)
                        else:
                            argv[-1] = 'native'
                    runner.assert_not_called()
                    assert {p.name: p.read_bytes() for p in out.iterdir()} == originals
        finally:
            os.chdir(old_cwd)


if __name__ == '__main__':
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(f'PASS {name}', flush=True)
