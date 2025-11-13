import pytest


def test_model_constructs_or_skips():
    try:
        from src.cmb.model import CosmologyAdvancedSampling
    except Exception:
        pytest.skip("Could not import CosmologyAdvancedSampling (missing deps)")

    # Construct with small parameters
    m = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)
    assert hasattr(m, "lmax")
    assert hasattr(m, "NSIDE")
    assert m.lmax == 8
    assert m.NSIDE == 2
