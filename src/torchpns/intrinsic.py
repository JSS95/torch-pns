import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "InverseIntrinsicPNS",
]


def _compute_rotation_matrix(v):
    a = np.zeros_like(v)
    a[-1] = 1.0
    b = v
    c = b - a * (np.dot(a, b))
    norm_c = np.linalg.norm(c)
    if norm_c > 1e-10:
        c /= norm_c
    else:
        c = np.zeros_like(c)

    A = np.outer(a, c) - np.outer(c, a)
    theta = np.arccos(v[-1])
    Id = np.eye(len(A))
    R = Id + np.sin(theta) * A + (np.cos(theta) - 1) * (np.outer(a, a) + np.outer(c, c))
    return R


class InverseIntrinsicPNS(nn.Module):
    """PyTorch module for inverse mapping of IntrinsicPNS.

    Allows inverse PNS to be part of a larger neural network model,
    with end-to-end training via backpropagation.

    Parameters
    ----------
    pns : skpns.IntrinsicPNS
        A fitted IntrinsicPNS model from scikit-pns.
    """

    def __init__(self, pns):
        super().__init__()
        self.n_components = pns.n_components
        self._n_features = pns._n_features

        # Register r_ as buffer
        self.register_buffer("r_", torch.from_numpy(np.array(pns.r_)).float())

        # Precompute scale factors for un-scaling Xi
        d = self._n_features - 1
        sin_rs = np.sin(pns.r_[:-1])
        scale_factors = np.ones(d)
        prod_sin_r = np.prod(sin_rs)
        for i in range(d - 1):
            scale_factors[i] = prod_sin_r
            prod_sin_r /= sin_rs[-i - 1]
        scale_factors[d - 1] = prod_sin_r
        self.register_buffer("scale_factors", torch.from_numpy(scale_factors).float())

        # Precompute offset for the lowest dimension (azimuthal angle)
        # v_[-1] is the last principal axis (2D)
        v_last = pns.v_[-1]
        offset = np.arctan2(v_last[1], v_last[0])
        self.register_buffer("offset", torch.tensor(offset).float())

        # Store v and R for each step of the reconstruction loop
        # The loop goes from i=0 to d-2
        # k = i + 1
        # We need v_[-1-k] and R(v_[-1-k])
        # We will store them as buffers with names v_inv_{i} and R_inv_{i}
        for i in range(d - 1):
            k = i + 1
            v = pns.v_[-1 - k]
            R = _compute_rotation_matrix(v)

            self.register_buffer(f"v_inv_{i}", torch.from_numpy(v).float())
            self.register_buffer(f"R_inv_{i}", torch.from_numpy(R).float())

    def forward(self, Xi):
        """
        Apply the inverse PNS transformation.

        Parameters
        ----------
        Xi : torch.Tensor
            Input tensor in the PNS embedded space (Euclidean).
            Shape: (N, n_components)

        Returns
        -------
        torch.Tensor
            Reconstructed tensor in the original manifold space (Sphere).
            Shape: (N, n_features)
        """
        N = Xi.shape[0]
        d = self._n_features - 1

        # Pad Xi if n_components < d
        if Xi.shape[1] < d:
            padding = torch.zeros(N, d - Xi.shape[1], device=Xi.device, dtype=Xi.dtype)
            Xi = torch.cat([Xi, padding], dim=1)

        # Un-scale Xi
        xi = Xi / self.scale_factors

        # Initialize x_dagger (lowest dimension)
        # xi[:, 0] is the azimuthal angle
        angle_0 = xi[:, 0] + self.offset

        # Convert to cartesian (2D)
        x_dagger = torch.stack([torch.cos(angle_0), torch.sin(angle_0)], dim=1)

        # Reconstruction loop
        for i in range(d - 1):
            k = i + 1

            # Retrieve parameters
            # r index: d - 1 - k
            r = self.r_[d - 1 - k]
            v = getattr(self, f"v_inv_{i}")
            R = getattr(self, f"R_inv_{i}")

            # Reconstruct to higher dimension sphere
            # vec = [sin(r) * x_dagger, cos(r)]
            sin_r = torch.sin(r)
            cos_r = torch.cos(r)

            # x_dagger: (N, dim)
            # vec: (N, dim+1)
            vec = torch.cat([sin_r * x_dagger, cos_r.expand(N, 1)], dim=1)

            # Rotate
            A = vec @ R

            # Inverse Projection
            # xi[:, k] is the residual angle
            xi_val = xi[:, k].unsqueeze(1)  # (N, 1)
            rho = xi_val + r

            # Note: skpns implementation uses np.sin(xi[:, k]) in _inv_proj call,
            # but _inv_proj logic implies xi is an angle.
            # We use the angle directly here as it is mathematically consistent.

            numerator = A * torch.sin(rho) - torch.sin(xi_val) * v.unsqueeze(0)
            x_dagger = numerator / sin_r

        return x_dagger
