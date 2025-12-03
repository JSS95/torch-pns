import numpy as np
import torch
from skpns import IntrinsicPNS
from skpns.util import circular_data

from torchpns import InverseIntrinsicPNS

np.random.seed(0)


def test_inverse_intrinsic_pns():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = circular_data()
    pns = IntrinsicPNS(n_components=2)
    X_transformed = pns.fit_transform(X)
    true_reconstruct = pns.inverse_transform(X_transformed)

    pns_torch = InverseIntrinsicPNS(pns).to(device)
    X_transformed_tensor = torch.from_numpy(X_transformed.copy()).float().to(device)

    assert torch.allclose(
        pns_torch.forward(X_transformed_tensor).cpu(),
        torch.from_numpy(true_reconstruct).float(),
        atol=1e-3,
    )
