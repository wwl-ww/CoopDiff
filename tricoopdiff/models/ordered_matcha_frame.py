"""Explicit continuous-domain Matcha frame variant; legacy adapter stays unchanged."""

import torch

from tricoopdiff.models.expert import FrozenMatcha
from tricoopdiff.runtime import sha256_file


class FrozenOrderedResidueMatcha(FrozenMatcha):
    """Use stable CA sequence indices, rejecting degenerate ordered frames."""

    frame_policy = 'ordered_residue_com_v1'

    def __init__(self, source_dir, weights_dir, device='cuda'):
        super().__init__(source_dir, weights_dir, device)
        self.provenance = {**self.provenance,
            'frame_policy': self.frame_policy,
            'coordinate_contract': 'protein_ordered_residue_com_frame_v1',
            'frame_adapter_sha256': sha256_file(__file__)}

    @staticmethod
    def protein_frame(positions, anchors, frame_policy='legacy_argmax_v1'):
        """World basis from fixed CA indices; differentiable on its valid domain."""
        origin = positions.mean(0)
        centered = positions - origin
        first, second = anchors
        first_axis = centered[first]
        first_norm = first_axis.norm()
        if frame_policy == 'ordered_residue_com_v1':
            if (not torch.isfinite(positions).all() or not torch.isfinite(centered).all()
                    or not torch.isfinite(first_norm.detach()) or first_norm.detach() < 1e-3):
                raise ValueError('ordered Matcha frame has degenerate first anchor; no fallback')
        e1 = first_axis / first_norm
        second_axis = centered[second] - (centered[second] @ e1) * e1
        second_norm = second_axis.norm()
        if frame_policy == 'ordered_residue_com_v1' and (
                not torch.isfinite(second_norm.detach()) or second_norm.detach() < 1e-3):
            raise ValueError('ordered Matcha frame has collinear anchors; no fallback')
        e2 = second_axis / second_norm
        return origin, torch.stack((e1, e2, torch.linalg.cross(e1, e2)), -1)

    @staticmethod
    def protein_anchors(positions, frame_policy='legacy_argmax_v1'):
        """Choose ordered indices without consulting geometry in the new policy."""
        if frame_policy == 'ordered_residue_com_v1':
            if len(positions) < 3:
                raise ValueError('ordered Matcha frame requires at least three residues')
            anchors = (0, 2 * len(positions) // 3)
            FrozenOrderedResidueMatcha.protein_frame(positions, anchors, frame_policy)
            return anchors
        if frame_policy != 'legacy_argmax_v1':
            raise ValueError(f'unknown Matcha frame policy: {frame_policy}')
        centered = positions - positions.mean(0)
        first = int(centered.square().sum(-1).argmax())
        axis = centered[first] / centered[first].norm()
        perpendicular = centered - (centered @ axis)[:, None] * axis
        second = int(perpendicular.square().sum(-1).argmax())
        if perpendicular[second].norm() < 1e-3:
            raise ValueError('protein cannot define a nondegenerate rigid reference frame')
        return first, second

    def prepare_case(self, e3_data, poi_data, ligand_data, protein_paths):
        # Parent preparation provides the unchanged sequence/chemistry pipeline.
        # Its legacy anchors are discarded; they never choose this variant's frame.
        prepared = list(super().prepare_case(e3_data, poi_data, ligand_data, protein_paths))
        for index in (0, 1):
            part = prepared[index]
            anchors = self.protein_anchors(part['coords'].reshape(-1, 3), self.frame_policy)
            prepared[index] = {**part, 'matcha': {**part['matcha'],
                'anchors': anchors, 'frame_policy': self.frame_policy}}
        return tuple(prepared)

    def read_pair(self, receptor, ligand, reverse_t, modulation=None):
        data = receptor['matcha']
        if data.get('frame_policy') != self.frame_policy:
            raise ValueError('Matcha protein metadata frame policy differs from expert identity')
        positions = receptor['coords'][0]
        expected_anchors = (0, 2 * len(positions) // 3)
        if tuple(data['anchors']) != expected_anchors:
            raise ValueError('Matcha ordered frame metadata has noncanonical residue anchors')
        # Rigid updates preserve nondegeneracy; validate external states too.
        self.protein_frame(positions, data['anchors'], self.frame_policy)
        return super().read_pair(receptor, ligand, reverse_t, modulation)
