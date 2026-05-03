"""
Tests for the chunked Pearson cross-correlation helpers in Analysis_Utils.

Key idea being tested: after row-normalising stim and resp separately,
the (n_neurons × n_features) Pearson cross-correlation is just Z_resp @ Z_stim.T.
The three execution paths (CPU fallback, stim-fits-on-GPU, 2-D tiling) must all
produce numerically identical results.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from Waven.Analysis_Utils import _row_normalize, _pearson_cross_corr_chunked


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_data(rng):
    """Small random stim/resp pair used across several tests."""
    T, n_features, n_neurons = 120, 15, 10
    stim = rng.standard_normal((T, n_features)).astype(np.float32)
    resp = rng.standard_normal((T, n_neurons)).astype(np.float32)
    return stim, resp


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _numpy_reference(stim, resp):
    """Pearson cross-correlation via numpy.corrcoef — used as ground truth."""
    n_features = stim.shape[1]
    combined = np.concatenate([stim.T, resp.T], axis=0)
    cc = np.corrcoef(combined)
    return cc[n_features:, :n_features].astype(np.float32)


def _normalised_tensors(stim, resp):
    Z_stim = _row_normalize(torch.as_tensor(stim.T, dtype=torch.float32))
    Z_resp = _row_normalize(torch.as_tensor(resp.T, dtype=torch.float32))
    return Z_stim, Z_resp


def _mock_props(total_memory):
    props = MagicMock()
    props.total_memory = total_memory
    return props


# ---------------------------------------------------------------------------
# _row_normalize
# ---------------------------------------------------------------------------

def test_row_normalize_mean_zero():
    mat = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    out = _row_normalize(mat)
    assert torch.allclose(out.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_row_normalize_unit_norm():
    mat = torch.randn(8, 50)
    out = _row_normalize(mat)
    assert torch.allclose(out.norm(dim=1), torch.ones(8), atol=1e-6)


def test_row_normalize_constant_row_no_nan():
    # A constant row has zero variance; the clamp in _row_normalize must
    # prevent division by zero and keep the output finite.
    mat = torch.ones(4, 20)
    out = _row_normalize(mat)
    assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# _pearson_cross_corr_chunked — three execution paths
# ---------------------------------------------------------------------------

def test_cpu_fallback_matches_numpy(small_data):
    """When CUDA is unavailable the function falls back to a plain matmul on CPU."""
    stim, resp = small_data
    Z_stim, Z_resp = _normalised_tensors(stim, resp)

    with patch("torch.cuda.is_available", return_value=False):
        result = _pearson_cross_corr_chunked(Z_stim, Z_resp, torch.device("cpu"))

    np.testing.assert_allclose(result, _numpy_reference(stim, resp), atol=1e-4)


def test_stim_fits_path_matches_numpy(small_data):
    """Stim-fits-on-GPU path: mock a GPU large enough that stim sits entirely
    on-device and only neurons are chunked."""
    stim, resp = small_data
    T, n_features = stim.shape
    Z_stim, Z_resp = _normalised_tensors(stim, resp)

    # Give mock GPU 4× the stim footprint so stim comfortably fits in half budget.
    mock_mem = n_features * T * 4 * 4

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=_mock_props(mock_mem)), \
         patch("torch.cuda.empty_cache"):
        # device='cpu' makes every .to(device) call a no-op, so the math
        # actually runs on CPU and remains verifiable.
        result = _pearson_cross_corr_chunked(Z_stim, Z_resp, torch.device("cpu"))

    np.testing.assert_allclose(result, _numpy_reference(stim, resp), atol=1e-4)


def test_2d_tile_path_matches_numpy(small_data):
    """2-D tiling path: mock a GPU too small to hold the full stim, forcing the
    square-tile branch."""
    stim, resp = small_data
    T, n_features = stim.shape
    Z_stim, Z_resp = _normalised_tensors(stim, resp)

    # Give mock GPU only 1/4 of the stim footprint — stim definitely doesn't fit.
    mock_mem = n_features * T * 4 // 4

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=_mock_props(mock_mem)), \
         patch("torch.cuda.empty_cache"):
        result = _pearson_cross_corr_chunked(Z_stim, Z_resp, torch.device("cpu"))

    np.testing.assert_allclose(result, _numpy_reference(stim, resp), atol=1e-4)


def test_all_paths_agree(small_data):
    """Sanity check: all three paths produce the same result for the same input."""
    stim, resp = small_data
    T, n_features = stim.shape
    Z_stim, Z_resp = _normalised_tensors(stim, resp)

    with patch("torch.cuda.is_available", return_value=False):
        cpu_result = _pearson_cross_corr_chunked(Z_stim, Z_resp, torch.device("cpu"))

    mock_mem_large = n_features * T * 4 * 4
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=_mock_props(mock_mem_large)), \
         patch("torch.cuda.empty_cache"):
        stim_fits_result = _pearson_cross_corr_chunked(Z_stim, Z_resp, torch.device("cpu"))

    mock_mem_small = n_features * T * 4 // 4
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=_mock_props(mock_mem_small)), \
         patch("torch.cuda.empty_cache"):
        tiled_result = _pearson_cross_corr_chunked(Z_stim, Z_resp, torch.device("cpu"))

    np.testing.assert_allclose(cpu_result, stim_fits_result, atol=1e-6)
    np.testing.assert_allclose(cpu_result, tiled_result, atol=1e-6)
