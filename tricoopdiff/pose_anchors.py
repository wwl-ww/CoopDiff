"""Joint position-orientation anchors for multi-hypothesis rigid placement."""

from dataclasses import dataclass

import torch

from .rigid import apply_rigid


def _normalize(vector, name, eps=1e-8):
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    if bool((norm <= eps).any()):
        raise ValueError(f"{name} contains a degenerate vector")
    return vector / norm


def surface_frame(normal, tangent):
    """Construct right-handed local frames with rows ``(x, y, normal)``."""
    if normal.shape != tangent.shape or normal.shape[-1] != 3:
        raise ValueError("normal and tangent must have matching [..., 3] shapes")
    z_axis = _normalize(normal, "normal")
    tangent = tangent - (tangent * z_axis).sum(-1, keepdim=True) * z_axis
    x_axis = _normalize(tangent, "tangent")
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    return torch.stack((x_axis, y_axis, z_axis), dim=-2)


class _ProperPolarRotation(torch.autograd.Function):
    """Differentiate the rotation, not ambiguous SVD singular vectors.

    The polar derivative uses sums of signed singular values, so repeated
    positive singular values (including an exact identity fit) are regular.
    """

    @staticmethod
    def forward(ctx, covariance):
        u, singular, vh = torch.linalg.svd(covariance.double())
        if bool((singular[..., 1] <= torch.finfo(covariance.dtype).eps
                 * singular[..., 0]).any()):
            raise ValueError('rigid fit is underdetermined for collinear coordinates')
        sign = torch.ones_like(singular)
        sign[..., -1] = torch.where(torch.det(u @ vh) < 0, -1., 1.)
        q = (u * sign.unsqueeze(-2)) @ vh
        ctx.save_for_backward(q, vh, singular * sign)
        ctx.input_dtype = covariance.dtype
        return q.to(covariance.dtype)

    @staticmethod
    def backward(ctx, gradient):
        q, vh, signed = ctx.saved_tensors
        a = q.transpose(-1, -2) @ gradient.double()
        skew = vh @ (a - a.transpose(-1, -2)) @ vh.transpose(-1, -2)
        denominator = signed.unsqueeze(-1) + signed.unsqueeze(-2)
        # Reflection/near-collinear fits can genuinely be ill-conditioned;
        # bound only that singular chart, not the ordinary polar derivative.
        floor = signed.abs().amax(-1, keepdim=True).unsqueeze(-1).clamp_min(1.) * 1e-8
        denominator = torch.where(denominator.abs() < floor,
                                  torch.where(denominator < 0, -floor, floor), denominator)
        k = vh.transpose(-1, -2) @ (skew / denominator) @ vh
        return (q @ k).to(ctx.input_dtype)


def proper_rigid_fit(source, target):
    """Return the proper least-squares row-coordinate transform source -> target."""
    if (source.shape != target.shape or source.ndim not in (2, 3)
            or source.shape[-1] != 3 or source.shape[-2] < 3):
        raise ValueError("source and target must be matching [..., N, 3], N >= 3")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("source and target must be finite")
    source_center = source.mean(dim=-2)
    target_center = target.mean(dim=-2)
    x = source - source_center.unsqueeze(-2)
    y = target - target_center.unsqueeze(-2)
    covariance = x.transpose(-1, -2) @ y
    rotation = _ProperPolarRotation.apply(covariance).transpose(-1, -2)
    translation = target_center - (
        rotation @ source_center.unsqueeze(-1)
    ).squeeze(-1)
    fitted = apply_rigid(source, rotation, translation)
    rmsd = (fitted - target).square().sum(-1).mean(-1).sqrt()
    return rotation, translation, rmsd


@dataclass(frozen=True)
class RigidPoseAnchors:
    """A set of proper rigid transforms using the row-vector convention."""

    rotations: torch.Tensor
    translations: torch.Tensor
    source_points: torch.Tensor
    target_points: torch.Tensor
    twists: torch.Tensor

    def __post_init__(self):
        count = self.rotations.shape[0]
        if self.rotations.shape != (count, 3, 3):
            raise ValueError("rotations must be [K, 3, 3]")
        for name in ("translations", "source_points", "target_points"):
            if getattr(self, name).shape != (count, 3):
                raise ValueError(f"{name} must be [K, 3]")
        if self.twists.shape != (count,):
            raise ValueError("twists must be [K]")
        identity = torch.eye(3, dtype=self.rotations.dtype,
                             device=self.rotations.device).expand(count, -1, -1)
        gram = self.rotations.transpose(-1, -2) @ self.rotations
        if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-5):
            raise ValueError("anchor rotations must be orthonormal")
        determinant = torch.linalg.det(self.rotations)
        if not torch.allclose(determinant, torch.ones_like(determinant),
                              atol=1e-5, rtol=1e-5):
            raise ValueError("anchor rotations must be proper (det=+1)")
        if not all(torch.isfinite(value).all() for value in (
                self.rotations, self.translations, self.source_points,
                self.target_points, self.twists)):
            raise ValueError("anchor tensors must be finite")

    def place(self, coordinates):
        """Apply every anchor transform to coordinates ``[N, 3]`` -> ``[K,N,3]``."""
        if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must be [N, 3]")
        return (torch.einsum("ni,kij->knj", coordinates, self.rotations)
                + self.translations[:, None, :])


def joint_surface_anchors(source_points, source_normals, source_tangents,
                          target_points, target_normals, target_tangents,
                          twists):
    """Create position+orientation hypotheses for paired surface anchors.

    Each source surface normal is aligned against the corresponding target
    surface normal. ``twists`` enumerate rotations around that contact normal,
    so one pocket pair can retain several plausible in-plane orientations.
    """
    inputs = (source_points, source_normals, source_tangents,
              target_points, target_normals, target_tangents)
    if any(value.ndim != 2 or value.shape[-1] != 3 for value in inputs):
        raise ValueError("surface anchor tensors must be [K, 3]")
    count = source_points.shape[0]
    if any(value.shape[0] != count for value in inputs):
        raise ValueError("surface anchor tensors must have equal K")
    if twists.ndim != 1 or twists.numel() == 0:
        raise ValueError("twists must be a non-empty [J] tensor")

    source_frames = surface_frame(source_normals, source_tangents)
    target_frames = surface_frame(-target_normals, target_tangents)
    target_x, target_y, target_z = target_frames.unbind(dim=-2)
    cosine = torch.cos(twists)[None, :, None]
    sine = torch.sin(twists)[None, :, None]
    twisted_x = cosine * target_x[:, None] + sine * target_y[:, None]
    twisted_y = -sine * target_x[:, None] + cosine * target_y[:, None]
    twisted_z = target_z[:, None].expand(-1, twists.numel(), -1)
    twisted_frames = torch.stack((twisted_x, twisted_y, twisted_z), dim=-2)
    rotations = source_frames[:, None].transpose(-1, -2) @ twisted_frames
    source = source_points[:, None].expand(-1, twists.numel(), -1)
    target = target_points[:, None].expand_as(source)
    translations = target - torch.einsum("kji,kjil->kjl", source, rotations)
    expanded_twists = twists[None].expand(count, -1)
    return RigidPoseAnchors(
        rotations=rotations.flatten(0, 1),
        translations=translations.flatten(0, 1),
        source_points=source.flatten(0, 1),
        target_points=target.flatten(0, 1),
        twists=expanded_twists.flatten(),
    )
