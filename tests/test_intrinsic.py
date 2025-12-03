import numpy as np
import torch
from skpns import IntrinsicPNS
from skpns.util import circular_data

from torchpns import InverseIntrinsicPNS

np.random.seed(0)


def test_inverse_intrinsic_pns():
    X = circular_data()
    pns = IntrinsicPNS(n_components=2)
    X_transformed = pns.fit_transform(X).astype(np.float32)

    pns_torch = InverseIntrinsicPNS(pns)
    assert torch.allclose(
        pns_torch.forward(torch.from_numpy(X_transformed).float()),
        torch.from_numpy(pns.inverse_transform(X_transformed)).float(),
        atol=1e-3,
    )
