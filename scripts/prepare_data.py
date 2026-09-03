"""
Phase 0 Data Preparation Scripts.

Three data sources:
1. General ternary complexes (PDB query → filter → deduplicate)
2. PROTAC benchmark (~50 structures, unified format)
3. Pseudo-ternary data generation (optional, if real data < 1000)

Output: data/ternary_train.json, data/ternary_val.json, data/protac_benchmark.json
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add parent to path for imports (when running from scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# PDB Query & Filter
# ---------------------------------------------------------------------------

PDB_QUERY = """
SELECT pdb_id, resolution, num_chains, num_ligands
WHERE
    num_protein_chains >= 2
    AND num_ligands >= 1
    AND ligand_heavy_atoms >= 20
    AND resolution <= 2.5
    AND experimental_method = 'X-RAY DIFFRACTION'
"""


def query_pdb_ternary(
    output_path: str = 'data/pdb_ternary_raw.json',
    min_resolution: float = 2.5,
    min_ligand_heavy_atoms: int = 20,
    min_interface_area: float = 50.0,
) -> List[Dict]:
    """
    Query PDB for potential ternary complex structures.

    Uses the RCSB Search API (https://search.rcsb.org/).

    Filter criteria:
    - Two protein chains
    - One ligand (≥20 heavy atoms)
    - Resolution < 2.5Å
    - Interface area > 50 Å² (computed post-hoc)

    NOTE: This script requires the `requests` package and internet access.
    Run on a machine with network access.
    """
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' package required. Install with: pip install requests")
        return []

    # RCSB Search API query
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "greater_or_equal",
                        "value": 2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_nonpolymer_entity_instance_count",
                        "operator": "greater_or_equal",
                        "value": 1,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": min_resolution,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                        "operator": "exact_match",
                        "value": "experimental",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 5000,
            },
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }

    print("Querying RCSB PDB for ternary complexes...")
    try:
        resp = requests.post(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            json=query,
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        pdb_ids = [entry["identifier"] for entry in result.get("result_set", [])]
        print(f"  Found {len(pdb_ids)} candidate PDB entries")
    except Exception as e:
        print(f"  Query failed: {e}")
        print("  Falling back to manual PDB list. Provide a text file with PDB IDs.")
        return []

    # Further process: download each PDB, check chains, interface area, etc.
    # This is a batch job — run on server with PDB mmCIF files
    results = []
    for pdb_id in pdb_ids:
        results.append({
            'pdb_id': pdb_id,
            'status': 'pending_processing',
        })

    # Save raw results
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} entries to {output_path}")

    return results


def download_mmcif_files(
    raw_list: List[Dict],
    pdb_dir: str = 'data/pdb_mmcif',
    base_url: str = 'https://files.rcsb.org/download',
) -> Dict[str, int]:
    """Download missing RCSB mmCIF files, retaining existing files on restart."""
    output_dir = Path(pdb_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {'downloaded': 0, 'existing': 0, 'failed': 0}
    for entry in raw_list:
        pdb_id = entry['pdb_id'].upper()
        destination = output_dir / f'{pdb_id}.cif'
        if destination.is_file() and destination.stat().st_size > 0:
            counts['existing'] += 1
            continue
        temporary = destination.with_suffix('.cif.part')
        try:
            urllib.request.urlretrieve(f'{base_url}/{pdb_id}.cif', temporary)
            temporary.replace(destination)
            counts['downloaded'] += 1
        except Exception as error:
            temporary.unlink(missing_ok=True)
            counts['failed'] += 1
            print(f'  Failed {pdb_id}: {error}')
    print(f'mmCIF download summary: {counts}')
    return counts


def filter_ternary_complexes(
    raw_list: List[Dict],
    pdb_dir: str = 'data/pdb_mmcif',
    output_path: str = 'data/ternary_filtered.json',
    min_ligand_heavy_atoms: int = 20,
) -> List[Dict]:
    """
    Filter raw PDB list to confirmed ternary complexes.

    For each PDB entry:
    1. Parse mmCIF to identify chains
    2. Verify ≥2 protein chains and ≥1 ligand
    3. Compute interface area between chains
    4. Check for bridge-like ligand topology (ligand contacts both chains)

    Uses BioPython for structure parsing.
    """
    try:
        from Bio.PDB import MMCIFParser
    except ImportError:
        print("ERROR: 'biopython' required. Install with: pip install biopython")
        return []

    parser = MMCIFParser(QUIET=True)
    confirmed = []

    for entry in raw_list:
        pdb_id = entry['pdb_id']
        cif_path = os.path.join(pdb_dir, f'{pdb_id}.cif')

        if not os.path.exists(cif_path):
            print(f"  Skipping {pdb_id}: mmCIF file not found")
            # Download via requests if not present
            continue

        try:
            structure = parser.get_structure(pdb_id, cif_path)
            model = structure[0]

            # Get protein chains and ligand
            chains = _get_protein_chains(model)
            ligands = _get_ligands(model, min_heavy_atoms=min_ligand_heavy_atoms)

            if len(chains) < 2:
                continue
            if len(ligands) < 1:
                continue

            # Search every chain-pair/ligand combination. Selecting only the first
            # two chains and first HET residue systematically misses biological
            # assemblies containing cofactors, waters, or symmetry mates.
            for chain_a, chain_b in combinations(chains, 2):
                interface_area = _compute_interface_area(chain_a, chain_b, cutoff=5.0)
                if interface_area < 50.0:
                    continue
                for ligand in ligands:
                    contacts_a = _count_contacts(ligand, chain_a, cutoff=5.0)
                    contacts_b = _count_contacts(ligand, chain_b, cutoff=5.0)
                    if contacts_a == 0 or contacts_b == 0:
                        continue
                    ligand_chain = ligand.get_parent().get_id()
                    hetflag, sequence_id, insertion_code = ligand.get_id()
                    confirmed.append({
                        'case': (
                            f'{pdb_id.upper()}_{chain_a.get_id()}_{chain_b.get_id()}_'
                            f'{ligand.resname}_{ligand_chain}_{sequence_id}'
                        ),
                        'pdb_id': pdb_id.upper(),
                        'chain_e3': chain_a.get_id(),
                        'chain_poi': chain_b.get_id(),
                        'ligand_chain': ligand_chain,
                        'ligand_resname': ligand.resname,
                        'ligand_hetflag': hetflag,
                        'ligand_sequence_id': int(sequence_id),
                        'ligand_insertion_code': insertion_code.strip(),
                        'ligand_heavy_atoms': _heavy_atom_count(ligand),
                        'interface_area': float(interface_area),
                        'contacts_e3': contacts_a,
                        'contacts_poi': contacts_b,
                    })

        except Exception as e:
            print(f"  Error processing {pdb_id}: {e}")
            continue

    print(f"Confirmed ternary complexes: {len(confirmed)} / {len(raw_list)}")

    with open(output_path, 'w') as f:
        json.dump(confirmed, f, indent=2)

    return confirmed


def materialize_ternary_complexes(
    complexes: List[Dict],
    pdb_dir: str = 'data/pdb_mmcif',
    output_dir: str = 'data/ternary_structures',
) -> Tuple[List[Dict], List[Dict]]:
    """Export each accepted chain pair and ligand in the training-entry layout."""
    from Bio.PDB import MMCIFParser, PDBIO, Select

    class ChainSelect(Select):
        def __init__(self, chain_id):
            self.chain_id = chain_id

        def accept_chain(self, chain):
            return int(chain.get_id() == self.chain_id)

        def accept_residue(self, residue):
            return int(residue.get_id()[0] == ' ')

    class LigandSelect(Select):
        def __init__(self, entry):
            self.entry = entry

        def accept_residue(self, residue):
            chain_id = residue.get_parent().get_id()
            hetflag, sequence_id, insertion_code = residue.get_id()
            return int(
                chain_id == self.entry['ligand_chain']
                and residue.resname == self.entry['ligand_resname']
                and int(sequence_id) == int(self.entry['ligand_sequence_id'])
                and insertion_code.strip() == self.entry.get('ligand_insertion_code', '')
            )

    parser = MMCIFParser(QUIET=True)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    entries, rejected = [], []
    parsed = {}
    for item in complexes:
        try:
            pdb_id = item['pdb_id'].upper()
            if pdb_id not in parsed:
                parsed[pdb_id] = parser.get_structure(
                    pdb_id, str(Path(pdb_dir) / f'{pdb_id}.cif')
                )
            structure = parsed[pdb_id]
            case_dir = root / item['case']
            case_dir.mkdir(parents=True, exist_ok=True)
            paths = {
                'protein1': case_dir / 'protein1.pdb',
                'protein2': case_dir / 'protein2.pdb',
                'ligand': case_dir / 'ligand.pdb',
            }
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(paths['protein1']), ChainSelect(item['chain_e3']))
            io.save(str(paths['protein2']), ChainSelect(item['chain_poi']))
            io.save(str(paths['ligand']), LigandSelect(item))
            if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
                raise RuntimeError('one or more exported files are empty')
            entry = dict(item)
            entry.update({key: _portable_path(path) for key, path in paths.items()})
            entries.append(entry)
        except Exception as error:
            rejected.append({'case': item.get('case'), 'reason': str(error)})
    print(f'Materialized {len(entries)} complexes; rejected {len(rejected)}')
    return entries, rejected


def split_ternary_entries(
    entries: List[Dict],
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Dict[str, List[Dict]]:
    """Deterministically split by PDB entry so one structure cannot leak."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError('val/test fractions must be non-negative and sum to < 1')
    groups = {}
    for entry in entries:
        groups.setdefault(entry['pdb_id'].upper(), []).append(entry)
    pdb_ids = sorted(groups)
    random.Random(seed).shuffle(pdb_ids)
    n_groups = len(pdb_ids)
    n_test = int(round(n_groups * test_fraction))
    n_val = int(round(n_groups * val_fraction))
    if n_groups >= 3 and test_fraction > 0:
        n_test = max(1, n_test)
    if n_groups - n_test >= 2 and val_fraction > 0:
        n_val = max(1, n_val)
    test_ids = set(pdb_ids[:n_test])
    val_ids = set(pdb_ids[n_test:n_test + n_val])
    result = {'train': [], 'val': [], 'test': []}
    for pdb_id, group in groups.items():
        split = 'test' if pdb_id in test_ids else 'val' if pdb_id in val_ids else 'train'
        result[split].extend(sorted(group, key=lambda item: item['case']))
    return result


def write_training_splits(splits: Dict[str, List[Dict]], output_dir: str = 'data'):
    """Write train-entry-compatible JSON indexes plus split metadata."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, entries in splits.items():
        path = output / f'ternary_{split}.json'
        with path.open('w', encoding='utf-8') as handle:
            json.dump({'entries': entries}, handle, indent=2)
        print(f'Wrote {len(entries)} entries to {path}')


# ---------------------------------------------------------------------------
# Sequence Identity Clustering (40% cutoff)
# ---------------------------------------------------------------------------

def cluster_by_sequence_identity(
    complexes: List[Dict],
    identity_cutoff: float = 0.40,
    output_path: str = 'data/ternary_clustered.json',
) -> List[Dict]:
    """
    Cluster ternary complexes by 40% sequence identity using MMseqs2 or CD-HIT.

    For now, uses a greedy pairwise method.
    For production, use MMseqs2 easy-cluster.
    """
    # Placeholder: in production, call MMseqs2
    # mmseqs easy-cluster fasta_input cluster_output tmp --min-seq-id 0.4

    # Greedy fallback: pick one representative per cluster
    # (requires pairwise sequence alignment — use BioPython)
    print(f"Clustering {len(complexes)} complexes at {identity_cutoff} identity...")
    print("  NOTE: Use MMseqs2 for production clustering. Greedy method is slow.")

    # Simplified: shuffle and take one per rough cluster
    # Full implementation requires FASTA extraction + MMseqs2
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(complexes))

    # For now, just return the shuffled list
    # TODO: Integrate MMseqs2 for proper clustering
    clustered = [complexes[i] for i in indices]

    with open(output_path, 'w') as f:
        json.dump(clustered, f, indent=2)

    return clustered


# ---------------------------------------------------------------------------
# PROTAC Benchmark Preparation
# ---------------------------------------------------------------------------

def prepare_protac_benchmark(
    raw_dir: str = 'data/protac_raw',
    output_path: str = 'data/protac_benchmark.json',
) -> List[Dict]:
    """
    Prepare PROTAC benchmark dataset in unified format.

    Each entry:
    {
        'name': str,
        'e3_type': 'VHL' | 'CRBN' | 'other',
        'poi': str (protein name),
        'warhead_e3': str (E3 binder SMILES),
        'warhead_poi': str (POI binder SMILES),
        'linker': str (linker SMILES),
        'protac_smiles': str (full PROTAC SMILES),
        'pdb_file': str (path to structure),
        'cooperativity_alpha': float | None,
        'dc50': float | None,
        'assay_type': str,
    }

    Split: by E3 type (avoid same-E3 data leakage between train/test)
    """
    entries = []

    if not os.path.exists(raw_dir):
        print(f"Raw PROTAC data directory not found: {raw_dir}")
        print("Please populate with PROTAC structures in PDB format.")
        print("Expected structure: {raw_dir}/{name}/{name}.pdb")
        return entries

    for name in os.listdir(raw_dir):
        entry_dir = os.path.join(raw_dir, name)
        if not os.path.isdir(entry_dir):
            continue

        pdb_file = os.path.join(entry_dir, f'{name}.pdb')
        if not os.path.exists(pdb_file):
            continue

        # Read metadata if available
        meta_file = os.path.join(entry_dir, 'metadata.json')
        metadata = {}
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                metadata = json.load(f)

        entries.append({
            'name': name,
            'e3_type': metadata.get('e3_type', 'unknown'),
            'poi': metadata.get('poi', 'unknown'),
            'warhead_e3': metadata.get('warhead_e3', ''),
            'warhead_poi': metadata.get('warhead_poi', ''),
            'linker': metadata.get('linker', ''),
            'protac_smiles': metadata.get('protac_smiles', ''),
            'pdb_file': pdb_file,
            'cooperativity_alpha': metadata.get('alpha'),
            'dc50': metadata.get('dc50'),
            'assay_type': metadata.get('assay_type', ''),
        })

    # Split by E3 type
    e3_types = set(e['e3_type'] for e in entries)
    train_entries, test_entries = [], []

    for e3_type in e3_types:
        type_entries = [e for e in entries if e['e3_type'] == e3_type]
        # Put at least one per E3 type in test set
        if len(type_entries) >= 3:
            n_test = max(1, len(type_entries) // 5)  # ~20% test
            test_entries.extend(type_entries[:n_test])
            train_entries.extend(type_entries[n_test:])
        else:
            train_entries.extend(type_entries)

    print(f"PROTAC benchmark: {len(train_entries)} train, {len(test_entries)} test")
    print(f"  E3 types: {e3_types}")

    output = {
        'train': _format_protac_entries(train_entries),
        'test': _format_protac_entries(test_entries),
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return entries


def _format_protac_entries(entries):
    """Format PROTAC entries into JSON-serializable dicts."""
    formatted = []
    for e in entries:
        formatted.append({
            'name': e['name'],
            'e3_type': e['e3_type'],
            'poi': e['poi'],
            'pdb_file': e['pdb_file'],
            'cooperativity_alpha': e['cooperativity_alpha'],
        })
    return formatted


# ---------------------------------------------------------------------------
# Pseudo-Ternary Data Generation
# ---------------------------------------------------------------------------

def generate_pseudo_ternary(
    protein_ligand_pairs: List[Tuple[str, str]],
    output_path: str = 'data/pseudo_ternary.json',
    n_samples: int = 1000,
    linker_pool: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Generate pseudo-ternary complexes by randomly pairing protein-ligand
    co-crystals and adding a virtual linker.

    This is a FALLBACK strategy if real ternary data < 1000.

    Steps:
    1. Randomly pair two protein-ligand co-crystal structures
    2. Align them via a virtual linker (linear alkane chain)
    3. Apply rigid-body adjustment
    4. Filter by physical plausibility (no clashes, reasonable distance)

    Args:
        protein_ligand_pairs: list of (protein_pdb, ligand_sdf) paths
        output_path: where to save the generated data
        n_samples: number of pseudo-complexes to generate
        linker_pool: optional list of linker SMILES to use
    """
    import random

    if linker_pool is None:
        # Simple PEG-like linkers
        linker_pool = [
            'CCOCCOCCOCC',      # tri-ethylene glycol
            'CCOCCOCC',         # di-ethylene glycol
            'CCOCCOCCOCCOCC',   # penta-ethylene glycol
            'CCCCCCCC',         # octane
            'CCCCCC',           # hexane
        ]

    rng = random.Random(42)
    generated = []

    if len(protein_ligand_pairs) < 2:
        print("Need at least 2 protein-ligand pairs for pseudo-ternary generation")
        return []

    for i in range(n_samples):
        # Randomly pick two different pairs
        pair1, pair2 = rng.sample(protein_ligand_pairs, 2)
        linker = rng.choice(linker_pool)

        generated.append({
            'id': f'pseudo_{i:04d}',
            'e3_pdb': pair1[0],
            'e3_ligand': pair1[1],
            'poi_pdb': pair2[0],
            'poi_ligand': pair2[1],
            'virtual_linker': linker,
            'status': 'pending_generation',
        })

    print(f"Generated {len(generated)} pseudo-ternary entries")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(generated, f, indent=2)

    return generated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_protein_chains(model):
    """Extract protein chains from a BioPython model."""
    chains = []
    for chain in model:
        # Check if chain contains amino acid residues
        residues = [r for r in chain if r.get_id()[0] == ' ']
        if len(residues) >= 10:  # minimum protein chain
            chains.append(chain)
    return chains


def _get_ligands(model, min_heavy_atoms=20):
    """Extract ligands (non-protein molecules) from a BioPython model."""
    ligands = []
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] != ' ':  # HETATM
                # Count heavy atoms
                heavy = [a for a in residue if a.element != 'H']
                if len(heavy) >= min_heavy_atoms:
                    ligands.append(residue)
    return ligands


def _heavy_atom_count(residue):
    """Count non-hydrogen atoms, tolerating blank mmCIF element fields."""
    return sum(1 for atom in residue if (atom.element or '').strip().upper() != 'H')


def _portable_path(path):
    """Store paths relative to the repository when possible."""
    resolved = Path(path).resolve()
    project_root = Path(__file__).resolve().parent.parent
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def _compute_interface_area(chain_a, chain_b, cutoff=5.0):
    """Compute approximate interface area between two chains."""
    from scipy.spatial.distance import cdist

    coords_a = np.array([a.get_coord() for r in chain_a for a in r])
    coords_b = np.array([a.get_coord() for r in chain_b for a in r])

    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0.0

    dist = cdist(coords_a, coords_b)
    contacts = (dist < cutoff).sum()
    # Rough area estimate: contacts × typical residue surface area
    return contacts * 25.0  # ~25 Å² per contact (approximate)


def _count_contacts(residue, chain, cutoff=5.0):
    """Count atomic contacts between a residue and a chain."""
    from scipy.spatial.distance import cdist

    coords_lig = np.array([a.get_coord() for a in residue])
    coords_chain = np.array([a.get_coord() for r in chain for a in r])

    if len(coords_lig) == 0 or len(coords_chain) == 0:
        return 0

    dist = cdist(coords_lig, coords_chain)
    return int((dist < cutoff).sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 0 data preparation for TriCoopDiff')
    parser.add_argument('--step', choices=[
        'query', 'download', 'filter', 'materialize', 'split', 'protac', 'pseudo', 'all',
    ],
                        default='all', help='Which preparation step to run')
    parser.add_argument('--pdb-dir', default='data/pdb_mmcif', help='Directory with PDB mmCIF files')
    parser.add_argument('--protac-dir', default='data/protac_raw', help='Directory with PROTAC structures')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--min-resolution', type=float, default=2.5)
    parser.add_argument('--min-ligand-heavy', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Process only the first N queried candidates (useful for a pilot run)',
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw_path = Path(args.output_dir) / 'pdb_ternary_raw.json'
    filtered_path = Path(args.output_dir) / 'ternary_filtered.json'
    materialized_path = Path(args.output_dir) / 'ternary_materialized.json'

    if args.step in ('query', 'all'):
        raw = query_pdb_ternary(
            output_path=os.path.join(args.output_dir, 'pdb_ternary_raw.json'),
            min_resolution=args.min_resolution,
            min_ligand_heavy_atoms=args.min_ligand_heavy,
        )
    else:
        raw = json.loads(raw_path.read_text()) if raw_path.is_file() else []
    process_raw = raw[:args.limit] if args.limit is not None else raw

    if args.step in ('download', 'all'):
        if not process_raw:
            raise SystemExit(f'No candidates found in {raw_path}; run --step query first')
        download_mmcif_files(process_raw, pdb_dir=args.pdb_dir)

    if args.step in ('filter', 'all'):
        if not process_raw:
            raise SystemExit(f'No candidates found in {raw_path}; run --step query first')
        filtered = filter_ternary_complexes(
            process_raw, pdb_dir=args.pdb_dir, output_path=str(filtered_path),
            min_ligand_heavy_atoms=args.min_ligand_heavy,
        )
    else:
        filtered = json.loads(filtered_path.read_text()) if filtered_path.is_file() else []

    if args.step in ('materialize', 'all'):
        if not filtered:
            raise SystemExit(f'No filtered complexes found in {filtered_path}')
        materialized, rejected = materialize_ternary_complexes(
            filtered, pdb_dir=args.pdb_dir,
            output_dir=str(Path(args.output_dir) / 'ternary_structures'),
        )
        materialized_path.write_text(json.dumps({
            'entries': materialized, 'rejected': rejected,
        }, indent=2), encoding='utf-8')
    elif materialized_path.is_file():
        materialized = json.loads(materialized_path.read_text()).get('entries', [])
    else:
        materialized = []

    if args.step in ('split', 'all'):
        if not materialized:
            raise SystemExit(f'No materialized complexes found in {materialized_path}')
        splits = split_ternary_entries(
            materialized, seed=args.seed,
            val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        )
        write_training_splits(splits, args.output_dir)

    if args.step in ('protac', 'all'):
        prepare_protac_benchmark(
            raw_dir=args.protac_dir,
            output_path=os.path.join(args.output_dir, 'protac_benchmark.json'),
        )

    print("\nPhase 0 data preparation complete.")
    print(f"Check {args.output_dir}/ for output files.")
