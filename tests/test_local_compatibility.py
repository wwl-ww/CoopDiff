"""Local support, symmetry and trainable cost-gradient correction contracts."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
from test_sdpi_flow import fixture
from tricoopdiff.geometry.rigid import rotation_exp
from tricoopdiff.models.local_compatibility import LocalCompatibilityCorrection, motion_jacobian


def setup(active=True):
    torch.manual_seed(31)
    flow, ld, e3, poi, _ = fixture()
    e3,poi = e3*.1,poi*.1
    ligand = ld['coords']
    coords = torch.cat((e3[0],poi[0],ligand[0]))
    types = torch.cat((torch.zeros(e3.shape[1]),torch.ones(poi.shape[1]),torch.full((ligand.shape[1],),2))).long()
    from tricoopdiff.diffdock.constants import lig_feature_dims
    features = torch.randn(len(coords),20,dtype=torch.double)
    features[types != 2, 0] = torch.randint(0,38,(int((types != 2).sum()),)).double()
    for column,size in enumerate(lig_feature_dims[0]):
        features[types == 2,column] = torch.randint(size,(int((types == 2).sum()),)).double()
    graph = {'node_coords':coords,'node_types':types,'node_features':features,
             'protein_local_geometry':torch.randn(e3.shape[1]+poi.shape[1],32,dtype=torch.double),
             'lig_edge_index':ld['edge_index'],'lig_edge_mask':ld['edge_mask'],'lig_mask_rotate':ld['mask_rotate']}
    model = LocalCompatibilityCorrection(20,hidden_dim=12).double()
    if active:
        for head in (model.pair_head,model.triplet_head):
            torch.nn.init.normal_(head[-1].weight,std=.1)
            torch.nn.init.constant_(head[-1].bias,.1)
    scales = flow.scales(e3,poi)
    return model,graph,scales,ld


def test_zero_initialization_has_first_update_readout_gradients():
    model,graph,scales,_ = setup(False)
    result = model(graph,scales)
    assert torch.equal(result['vector'],torch.zeros_like(result['vector']))
    target = torch.linspace(-.2,.3,len(result['vector']),dtype=torch.double)
    (result['vector']-target).square().sum().backward()
    for head in (model.pair_head,model.triplet_head):
        gradient = head[-1].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm()>0


def test_se3_equivariance_and_descent_bound():
    model,graph,scales,_ = setup()
    result = model(graph,scales)
    rotation = rotation_exp(torch.tensor([.4,-.8,.3],dtype=torch.double))
    changed = {**graph,'node_coords':graph['node_coords']@rotation.T+torch.tensor([9.,-2.,4.])}
    rotated = model(changed,scales)
    expected = result['vector'].clone()
    expected[:12] = (expected[:12].reshape(4,3)@rotation.T).flatten()
    assert torch.allclose(rotated['vector'],expected,atol=1e-9,rtol=1e-6)
    assert torch.allclose(rotated['local_cost'],result['local_cost'],atol=1e-10)
    assert result['diagnostics']['cost_directional_derivative']<=1e-12
    assert result['vector'][:12].reshape(4,3).norm(dim=-1).max()<.25
    assert result['vector'][12:].abs().max()<.25


def test_far_field_and_missing_partner_are_exact_zero():
    model,graph,scales,_ = setup()
    coords = graph['node_coords'].clone()
    coords[graph['node_types']==1] += 100
    far = {**graph,'node_coords':coords}
    pair,triplet = model.costs(far)
    assert pair==0 and triplet==0
    output = model(far,scales)
    assert torch.equal(output['vector'],torch.zeros_like(output['vector']))
    output['vector'].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_no_grad_inference_and_second_order_training_and_no_torsion():
    model,graph,scales,_ = setup()
    coords = graph['node_coords'].clone().requires_grad_(True)
    graph = {**graph,'node_coords':coords}
    output = model(graph,scales,no_torsion=True)
    assert output['vector'].shape==(12,)
    output['vector'].square().sum().backward()
    assert coords.grad is not None and torch.isfinite(coords.grad).all() and coords.grad.norm()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    expected = output['vector'].detach()
    with torch.no_grad():
        actual = model(graph,scales,no_torsion=True)['vector']
    assert not actual.requires_grad and torch.allclose(actual,expected)
    with torch.inference_mode():
        inference = model(graph,scales,no_torsion=True)['vector']
    assert torch.allclose(inference,expected)


def test_projection_jacobian_matches_actual_symmetric_update():
    model,graph,scales,ld = setup()
    from tricoopdiff.sdpi_flow import SDPIFlow, INDEPENDENT_RIGID_MODE, SYMMETRIC_TORSION_MODE
    types,coords = graph['node_types'],graph['node_coords']
    jacobian = motion_jacobian(coords,types,graph,scales)
    vector = torch.randn(jacobian.shape[-1],dtype=torch.double)*.1
    state = (coords[types==2][None],coords[types==1][None],torch.eye(3,dtype=torch.double)[None])
    def step(sign):
        velocity={'rigid_output_mode':INDEPENDENT_RIGID_MODE,'torsion_update_mode':SYMMETRIC_TORSION_MODE,'torsion':sign*vector[None,12:]*torch.pi}
        for i,name in enumerate(('e3_ligand','poi_ligand')):
            velocity[name]={'tr':sign*vector[None,i*6:i*6+3]*scales[i],'rot':sign*vector[None,i*6+3:i*6+6]*torch.pi}
        return SDPIFlow.differentiable_step(*state,velocity,torch.tensor([1e-5],dtype=torch.double),ld)
    plus,minus=step(1),step(-1)
    for i,component in enumerate((2,1)):
        measured=(minus[i][0]-plus[i][0])/(2e-5)
        expected=torch.einsum('ncd,d->nc',jacobian[types==component],vector)
        assert torch.allclose(measured,expected,atol=1e-8,rtol=1e-6)



def test_executed_small_correction_decreases_compatibility_cost():
    model,graph,scales,ld = setup()
    from tricoopdiff.sdpi_flow import SDPIFlow, INDEPENDENT_RIGID_MODE, SYMMETRIC_TORSION_MODE
    output = model(graph,scales)
    vector = output['vector']
    assert vector.norm()>0
    types,coords = graph['node_types'],graph['node_coords']
    state = (coords[types==2][None],coords[types==1][None],torch.eye(3,dtype=torch.double)[None])
    velocity = {'rigid_output_mode':INDEPENDENT_RIGID_MODE,'torsion_update_mode':SYMMETRIC_TORSION_MODE,
                'torsion':vector[None,12:]*torch.pi}
    for i,name in enumerate(('e3_ligand','poi_ligand')):
        velocity[name]={'tr':vector[None,i*6:i*6+3]*scales[i],
                        'rot':vector[None,i*6+3:i*6+6]*torch.pi}
    next_state = SDPIFlow.differentiable_step(*state,velocity,torch.tensor([1e-3],dtype=torch.double),ld)
    moved = coords.clone()
    moved[types==2],moved[types==1] = next_state[0][0],next_state[1][0]
    pair,triplet = model.costs(graph,moved)
    assert pair+triplet<output['local_cost'], (pair+triplet,output['local_cost'])


def test_flat_far_neighborhood_second_coordinate_derivatives_are_finite():
    model,graph,scales,_ = setup()
    coords = graph['node_coords'].clone()
    coords[graph['node_types']==1] += 100
    coords.requires_grad_(True)
    result = model({**graph,'node_coords':coords},scales)
    first = torch.autograd.grad(result['vector'].sum(),coords,create_graph=True)[0]
    second = torch.autograd.grad(first.sum(),coords)[0]
    assert torch.equal(first,torch.zeros_like(first))
    assert torch.equal(second,torch.zeros_like(second))



def test_covalent_bridge_connects_spatially_separated_binding_ends():
    model,graph,_,_ = setup()
    types = graph['node_types']
    count = int((types==2).sum())
    assert count >= 6
    coords = graph['node_coords'].clone()
    coords[types==2] = torch.stack((torch.linspace(0,20,count),torch.zeros(count),torch.zeros(count)),-1).double()
    coords[types==0] = coords.new_tensor([0.,1.,0.])
    coords[types==1] = coords.new_tensor([20.,1.,0.])
    edge = torch.stack((torch.arange(count-1),torch.arange(1,count)))
    changed = {**graph,'node_coords':coords,'lig_edge_index':edge}
    # A constant positive coefficient isolates the support/bridge condition.
    with torch.no_grad():
        model.triplet_head[-1].weight.zero_()
        model.triplet_head[-1].bias.fill_(1.)
    a,_ = model._radial(coords[types==2],coords[types==0])
    b,_ = model._radial(coords[types==2],coords[types==1])
    assert torch.equal(a.sum(-1)*b.sum(-1),torch.zeros(count,dtype=torch.double))
    pair,bridged = model.costs(changed)
    assert pair==0 and bridged>0, bridged
    # Split covalent components between the two disjoint contact patches.
    keep = edge[0] != count//2-1
    disconnected = {**changed,'lig_edge_index':edge[:,keep]}
    assert model.costs(disconnected)[1]==0
    absent = coords.clone();absent[types==1] += 100
    assert model.costs({**changed,'node_coords':absent})[1]==0


if __name__=='__main__':
    tests=[value for name,value in globals().copy().items() if name.startswith('test_') and callable(value)]
    for test in tests:test()
    print(f'{len(tests)} local compatibility tests passed')
