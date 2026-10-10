"""Reusable synthetic inputs; no test module imports another test module."""

def ligand_input(smiles="CCCOCC"):
    """Build a deterministic, featurized ligand without external assets."""
    from rdkit import Chem
    from torch_geometric.data import HeteroData
    from tricoopdiff.diffdock.datasets.process_mols import get_lig_graph_with_matching
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=3) == 0
    graph = HeteroData()
    get_lig_graph_with_matching(mol, graph, popsize=None, maxiter=None,
                               matching=False, keep_original=False,
                               num_conformers=1, remove_hs=True)
    return {
        "mol": Chem.RemoveHs(mol).ToBinary(),
        "coords": graph['ligand'].pos.unsqueeze(0),
        "edge_index": graph['ligand', 'ligand'].edge_index,
        "edge_mask": graph['ligand'].edge_mask,
        "mask_rotate": graph['ligand'].mask_rotate,
    }


def flow_example(smiles='CCCOCC', seed=12):
    """Build float64 ligand/protein inputs and their SDPI coupling."""
    import numpy as np
    import torch

    from tricoopdiff.sdpi_flow import SDPIFlow

    np.random.seed(seed)
    torch.manual_seed(seed)
    data = ligand_input(smiles)
    data['coords'] = data['coords'].double()
    a = torch.randn(1, 20, 3, dtype=torch.float64) * 10
    b = torch.randn(1, 17, 3, dtype=torch.float64) * 8 + 7
    flow = SDPIFlow()
    coupling = flow.sample_coupling(a, b, data)
    return flow, data, a, b, coupling


def chemistry_example(root):
    """Write an atom-only ligand and its offline CCD reference."""
    import numpy as np
    from rdkit import Chem

    molecule = Chem.MolFromSmiles('CC(=O)[O-]')
    conformer = Chem.Conformer(4)
    coordinates = np.array([[1., 2., 3.], [2.4, 2., 3.],
                            [3., 3., 3.], [3., 1., 3.]])
    for index, (atom, name) in enumerate(zip(molecule.GetAtoms(), ('C2', 'C1', 'O1', 'O2'))):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(name.rjust(4))
        info.SetResidueName('TST')
        info.SetResidueNumber(1)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
        conformer.SetAtomPosition(index, coordinates[index])
    molecule.AddConformer(conformer)
    pdb = root / 'ligand.pdb'
    # Match the atom-only export that caused loss of bond-order information.
    pdb.write_text('\n'.join(line for line in Chem.MolToPDBBlock(molecule).splitlines()
                              if line.startswith('HETATM')) + '\nEND\n')
    ccd = root / 'TST.cif'
    ccd.write_text('''data_TST
_chem_comp.id TST
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
O2 O -1
C1 C 0
C2 C 0
O1 O 0
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
C1 O1 DOUB
C1 C2 SING
C1 O2 SING
''')
    return pdb, ccd, coordinates


def run_tests(path):
    """Keep documented direct commands while using one fixture-aware runner."""
    import sys
    import pytest

    raise SystemExit(pytest.main([path, "-q", *sys.argv[1:]]))


def distributed_gloo_worker(rank, rendezvous, result_dir):
    import torch
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist
    from tricoopdiff.distributed import GradientSynchronizer
    from tricoopdiff.train import TriCoopDiffTrainer

    class ToyTrainer(TriCoopDiffTrainer):
        def __init__(self):
            self.model = torch.nn.Linear(1, 1, bias=False)
            self.model.weight.data.fill_(0.)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.1)
            self.global_step = self.samples_seen = 0

        def compute_loss(self, value):
            return {'loss_total': self.model.weight.sum() * value}

    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        sync = GradientSynchronizer('cpu')
        parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
        parameters[0].grad = torch.tensor([1. if rank == 0 else 3.])
        if rank == 0:
            parameters[1].grad = torch.tensor([4.])
        count = sync(parameters, 1 if rank == 0 else 3)
        assert count == 4
        assert torch.equal(parameters[0].grad, torch.tensor([2.5]))
        assert torch.equal(parameters[1].grad, torch.tensor([1.]))
        assert parameters[2].grad is None, 'globally unused parameters must skip AdamW decay'
        trainer = ToyTrainer()
        trainer.gradient_sync = sync
        trainer.distributed_error_check = sync.check_error
        results = trainer.train_batch([{'value': .2 if rank == 0 else .4}] * (rank + 1))
        assert trainer.global_step == 1 and trainer.samples_seen == 3
        assert results[0]['optimizer_batch_size'] == 3
        assert torch.allclose(trainer.model.weight, torch.tensor([[-1. / 30.]]))
        before = trainer.model.weight.detach().clone()
        rng = torch.get_rng_state().clone()
        try:
            trainer.train_batch([{'value': float('nan') if rank == 0 else .3}])
        except RuntimeError as error:
            assert 'distributed update aborted' in str(error)
        else:
            raise AssertionError('one-rank nonfinite failure must abort every rank')
        assert trainer.samples_seen == 3 and trainer.global_step == 1
        assert torch.equal(before, trainer.model.weight)
        assert torch.equal(rng, torch.get_rng_state())
        assert trainer.model.weight.grad is None
        Path(result_dir, str(rank)).write_text('ok')
    finally:
        dist.destroy_process_group()
