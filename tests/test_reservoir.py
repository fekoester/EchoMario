import numpy as np

from echomario.reservoirs.esn import ESNConfig, IdentityReservoir, SparseESN, load_reservoir_from_state_dict


def test_sparse_reservoir_shape():
    r = SparseESN(ESNConfig(input_dim=5, size=100, sparsity=0.05))
    x = r.step(np.ones(5, dtype=np.float32))
    assert x.shape == (100,)
    assert np.isfinite(x).all()


def test_identity_reservoir_roundtrip():
    r = IdentityReservoir(input_dim=5)
    x = r.step(np.arange(5, dtype=np.float32))
    assert x.shape == (5,)
    assert np.allclose(x, np.arange(5, dtype=np.float32))

    restored = load_reservoir_from_state_dict(r.state_dict())
    y = restored.step(np.ones(5, dtype=np.float32))
    assert y.shape == (5,)
