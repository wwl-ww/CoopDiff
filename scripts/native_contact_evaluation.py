"""Native protein-contact validity annotations shared by small-sample evaluators."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from tricoopdiff.runtime import sha256_file
from tricoopdiff.metrics.dockq import _dockq_protein_records


def native_contact_preflight(context):
    """Use the same normalized native records and official 5A fnat executable."""
    native_path = Path(context['native_complex_pdb'])
    native = [line for line in native_path.read_text().splitlines() if line.startswith('ATOM')]
    native, _ = _dockq_protein_records(native, native)
    chains = sorted({line[21] for line in native})
    if len(chains)!=2:
        raise ValueError('native contact audit requires exactly two protein chains')
    def coords(chain):
        return np.asarray([[float(line[30:38]),float(line[38:46]),float(line[46:54])]
                           for line in native if line[21]==chain])
    from scipy.spatial.distance import cdist
    distances = cdist(coords(chains[0]),coords(chains[1]))
    deep = Path(context['deepternary_root'])
    sys.path.insert(0,str(deep)) if str(deep) not in sys.path else None
    from DockQ.dockq_util import parse_fnat
    with tempfile.TemporaryDirectory(prefix='coupled-native-contacts-') as directory:
        normalized = Path(directory)/'native.pdb'
        normalized.write_text('\n'.join(native)+'\n')
        result = subprocess.run([str(deep/'DockQ/fnat'),str(normalized),str(normalized),'5','-all'],
                                check=True,capture_output=True,text=True)
        fnat,correct,total,*_ = parse_fnat(result.stdout)
    if total<0:
        raise ValueError('official fnat did not return native contact count')
    return {'native_contact_count_5A':total, 'native_self_fnat':fnat,
        'zero_native_contacts':total==0,
        'native_min_protein_atom_distance_angstrom':float(distances.min()),
        'native_atom_pairs_below5A':int((distances<5).sum()),
        'native_complex_sha256':sha256_file(native_path),
        'contact_protocol':'official DockQ/fnat native native 5 -all; same normalized ATOM records'}


def contact_strata(records):
    def summarize(rows):
        return {'count':len(rows),'cases':[row['case'] for row in rows],
            'dockq_mean':sum(row['dockq'] for row in rows)/len(rows) if rows else None,
            'dockq_above_023':sum(row['dockq']>.23 for row in rows),
            'ligand_crystal_rmsd_mean':sum(row['ligand_crystal_rmsd_angstrom'] for row in rows)/len(rows) if rows else None,
            'ligand_reachable_rmsd_mean':sum(row['ligand_target_rmsd_angstrom'] for row in rows)/len(rows) if rows else None}
    return {'valid_pp_native_contact':summarize([row for row in records if not row['zero_native_contacts']]),
            'degenerate_cases':summarize([row for row in records if row['zero_native_contacts']])}
