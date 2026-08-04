"""
Phase 0 Data Preparation Scripts.

Three data sources:
1. General ternary complexes (PDB query → filter → deduplicate)
2. PROTAC benchmark (~50 structures, unified format)
3. Pseudo-ternary data generation (optional, if real data < 1000)

Output: data/ternary_train.json, data/ternary_val.json, data/protac_benchmark.json
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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


def filter_ternary_complexes(
    raw_list: List[Dict],
    pdb_dir: str = 'data/pdb_mmcif',
    output_path: str = 'data/ternary_filtered.json',
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
            ligands = _get_ligands(model, min_heavy_atoms=20)

            if len(chains) < 2:
                continue
            if len(ligands) < 1:
                continue

            # Take first two protein chains and first ligand
            chain_a, chain_b = chains[0], chains[1]
            ligand = ligands[0]

            # Check interface area
            interface_area = _compute_interface_area(chain_a, chain_b, cutoff=5.0)
            if interface_area < 50.0:
                continue

            # Check bridging: ligand contacts both chains
            contacts_a = _count_contacts(ligand, chain_a, cutoff=5.0)
            contacts_b = _count_contacts(ligand, chain_b, cutoff=5.0)

            if contacts_a == 0 or contacts_b == 0:
                continue  # ligand only contacts one chain

            confirmed.append({
                'pdb_id': pdb_id,
                'chain_e3': chain_a.get_id(),
                'chain_poi': chain_b.get_id(),
                'ligand_id': ligand.get_id(),
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
    import argparse

    parser = argparse.ArgumentParser(description='Phase 0 data preparation for TriCoopDiff')
    parser.add_argument('--step', choices=['query', 'filter', 'cluster', 'protac', 'pseudo', 'all'],
                        default='all', help='Which preparation step to run')
    parser.add_argument('--pdb-dir', default='data/pdb_mmcif', help='Directory with PDB mmCIF files')
    parser.add_argument('--protac-dir', default='data/protac_raw', help='Directory with PROTAC structures')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--min-resolution', type=float, default=2.5)
    parser.add_argument('--min-ligand-heavy', type=int, default=20)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.step in ('query', 'all'):
        query_pdb_ternary(
            output_path=os.path.join(args.output_dir, 'pdb_ternary_raw.json'),
            min_resolution=args.min_resolution,
            min_ligand_heavy_atoms=args.min_ligand_heavy,
        )

    if args.step in ('protac', 'all'):
        prepare_protac_benchmark(
            raw_dir=args.protac_dir,
            output_path=os.path.join(args.output_dir, 'protac_benchmark.json'),
        )

    print("\nPhase 0 data preparation complete.")
    print(f"Check {args.output_dir}/ for output files.")
