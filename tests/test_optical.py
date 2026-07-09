import numpy as np
import pytest
import xarray as xr

import artist


def _two_layer_dataset(extra):
    base = {
        "pres": (("height", "ncells"), [[100000.0], [90000.0]]),
        "temp": (("height", "ncells"), [[250.0], [260.0]]),
        "z_ifc": (("height_2", "ncells"), [[30.0], [20.0], [0.0]]),
    }
    base.update(extra)
    return xr.Dataset(base, coords={"height": [0, 1], "height_2": [0, 1, 2], "ncells": [0]})


def test_vmr_to_du_uses_pressure_temperature_and_layer_thickness():
    ds = _two_layer_dataset({"so2_vmr": (("height", "ncells"), [[1e-9], [2e-9]])})

    result = ds.art.vmr_to_du("so2_vmr")
    dz = xr.DataArray([[10.0], [20.0]], dims=("height", "ncells"))
    expected = ((ds.so2_vmr * ds.pres / (8.314472 * ds.temp)) * dz)
    expected = (expected * 6.02214076e23 / 2.69e20).sum("height")

    np.testing.assert_allclose(result.values, expected.values)


def test_density_is_mass_weighted_by_component_defaults():
    ds = xr.Dataset(
        {
            "ash_mixed_acc": ("x", [2.0]),
            "so4_mixed_acc": ("x", [1.0]),
            "h2o_mixed_acc": ("x", [1.0]),
        }
    )

    ash_density = ds.art.density(["ash_mixed_acc"])
    mixed_density = ds.art.density(["so4_mixed_acc", "h2o_mixed_acc"])

    assert ash_density.item() == pytest.approx(2650.0)
    assert mixed_density.item() == pytest.approx((1800.0 + 1000.0) / 2.0)


def test_aod_uses_insoluble_ash_and_layer_thickness():
    ds = _two_layer_dataset(
        {
            "rho": (("height", "ncells"), [[1.0], [1.0]]),
            "ash_insol_acc": (("height", "ncells"), [[2.0], [3.0]]),
            "ash_insol_coa": (("height", "ncells"), [[4.0], [5.0]]),
        }
    )

    result = ds.art.aod(532)
    dz = np.array([[10.0], [20.0]])
    expected_ext = (1.60644 * ds.ash_insol_acc.values + 0.15129 * ds.ash_insol_coa.values) * 1e-6

    np.testing.assert_allclose(result.values, expected_ext * dz)


def test_sulfate_aod_applies_script_thresholds():
    ds = _two_layer_dataset(
        {
            "rho": (("height", "ncells"), [[1.0], [1.0]]),
            "so4_sol_ait": (("height", "ncells"), [[2e-3], [1e-4]]),
            "so4_sol_acc": (("height", "ncells"), [[0.2], [0.05]]),
        }
    )

    result = ds.art.sulfate_aod(532)
    expected_first = (0.24575 * 2e-3 + 3.06857 * 0.2) * 1e-6 * 10.0

    assert result.values[0, 0] == pytest.approx(expected_first)
    assert result.values[1, 0] == pytest.approx(0.0)
    assert ds.art.saod(532).identical(result)


def test_ssa_532_is_mass_weighted_and_rejects_other_wavelengths():
    ds = xr.Dataset(
        {
            "ash_insol_acc": ("x", [1.0]),
            "ash_insol_coa": ("x", [1.0]),
            "ash_mixed_acc": ("x", [1.0]),
            "ash_mixed_coa": ("x", [1.0]),
            "so4_mixed_acc": ("x", [1.0]),
            "so4_mixed_coa": ("x", [1.0]),
            "so4_sol_ait": ("x", [1.0]),
            "so4_sol_acc": ("x", [1.0]),
        }
    )

    result = ds.art.ssa(532)
    expected = (
        0.98669
        + 0.90148
        + 0.99619 * 2.0
        + 0.93797 * 2.0
        + 1.0
        + 1.0
    ) / 8.0

    assert result.item() == pytest.approx(expected)
    with pytest.raises(ValueError, match="SSA is currently implemented only for 532 nm"):
        ds.art.ssa(355)
