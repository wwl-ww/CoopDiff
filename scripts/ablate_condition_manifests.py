"""Create explicit deploy-manifest feature ablations for mechanism audits."""

import argparse
import json
from pathlib import Path


MODES = ('role_only', 'pocket_only', 'role_pocket')
ROLE_FIELDS = {'fragment_smiles', 'atom_map', 'atom_maps', 'mapping_method'}
POCKET_FIELDS = {'pocket_probabilities'}
def ablate_side(side, mode):
    if side is None:
        return None
    keep = {'provenance', 'evidence_id', 'confidence', 'audit_only'}
    if mode in ('role_only', 'role_pocket'):
        keep |= ROLE_FIELDS
    if mode in ('pocket_only', 'role_pocket'):
        keep |= POCKET_FIELDS
    result = {key: value for key, value in side.items() if key in keep}
    result['evidence_id'] = f"{side['evidence_id']}#ablation:{mode}"
    if mode == 'pocket_only' and result.get('pocket_probabilities') is None:
        return None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--cases', nargs='*')
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        parser.error('choose a new output directory')
    sources = sorted(args.manifest_dir.glob('*.json'))
    if args.cases:
        wanted = set(args.cases)
        sources = [path for path in sources if path.stem in wanted]
        missing = wanted - {path.stem for path in sources}
        if missing:
            parser.error(f'missing manifests: {sorted(missing)}')
    sources = [path for path in sources if path.name != 'report.json']
    args.out_dir.mkdir(parents=True)
    written = []
    for source in sources:
        payload = json.loads(source.read_text())
        payload['sides'] = [ablate_side(side, args.mode) for side in payload['sides']]
        destination = args.out_dir / source.name
        destination.write_text(json.dumps(payload, indent=2) + '\n')
        written.append(payload['case'])
    report = {
        'protocol': 'condition_manifest_feature_ablation_v1',
        'mode': args.mode,
        'source': str(args.manifest_dir.resolve()),
        'cases': written,
        'native_ternary_geometry_read': False,
        'deployment_recommendation': False,
    }
    (args.out_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'mode': args.mode, 'cases': len(written)}, indent=2))


if __name__ == '__main__':
    main()
