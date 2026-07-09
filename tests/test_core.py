import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import artist
from artist.utils import add_grid, distance


def _write_icon_grid(path):
    xr.Dataset(
        {
            "clon": ("ncells", np.deg2rad([0.0, 1.0, 2.0, 3.0])),
            "clat": ("ncells", np.deg2rad([0.0, 0.0, 1.0, 1.0])),
            "clon_vertices": (
                ("ncells", "vertices"),
                np.deg2rad(
                    [
                        [-0.2, 0.2, 0.0],
                        [0.8, 1.2, 1.0],
                        [1.8, 2.2, 2.0],
                        [2.8, 3.2, 3.0],
                    ]
                ),
            ),
            "clat_vertices": (
                ("ncells", "vertices"),
                np.deg2rad(
                    [
                        [-0.2, -0.2, 0.2],
                        [-0.2, -0.2, 0.2],
                        [0.8, 0.8, 1.2],
                        [0.8, 0.8, 1.2],
                    ]
                ),
            ),
        }
    ).to_netcdf(path, engine="scipy")


def test_distance_and_add_grid(tmp_path):
    assert distance([0, 0], [0, 0]) == pytest.approx(0.0)
    assert distance([0, 0], [1, 0]) == pytest.approx(111.19, rel=1e-3)

    gridfile = tmp_path / "grid.nc"
    _write_icon_grid(gridfile)
    grid, vlon, vlat, clon, clat = add_grid(gridfile, ltranslon=False)

    assert clon.values.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert clat.values.tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert vlon.shape == (4, 3)
    assert vlat.shape == (4, 3)
    grid.close()


def test_icon_accessor_add_grid_selection_and_lookup(tmp_path):
    gridfile = tmp_path / "grid.nc"
    _write_icon_grid(gridfile)
    ds = xr.Dataset(
        {
            "ash_mixed_acc": ("ncells", [1.0, 2.0, 3.0, 4.0]),
            "temp": ("ncells", [280.0, 281.0, 282.0, 283.0]),
            "z_ifc": (("height_2", "ncells"), [[100.0, 100.0, 100.0, 100.0], [80.0, 70.0, 60.0, 50.0]]),
        },
        coords={"height_2": [0, 1]},
    )

    grid = ds.icon.add_grid(gridfile)
    grid.close()

    assert ds.attrs["_artist_gridfile"] == str(gridfile)
    assert ds.coords["ncells"].attrs["_artist_gridfile"] == str(gridfile)
    assert ds["ash_mixed_acc"].attrs["_artist_gridfile"] == str(gridfile)
    assert "clon" in ds.coords
    assert "clat_vertices" in ds.coords
    assert ds.icon.look_up("ash") == ["ash_mixed_acc"]
    assert ds.icon.nearest_gridpoints([1.1, 0.1]).item() == 1

    selected = ds.icon.sellonlat(0.5, 2.5, -0.5, 1.5)
    assert selected.sizes["ncells"] == 2

    dz = ds.icon.get_dz()
    assert dz.dims == ("height", "ncells")
    np.testing.assert_allclose(dz.values, [[20.0, 30.0, 40.0, 50.0]])


def test_dataarray_regrid_and_tri_plot(tmp_path):
    gridfile = tmp_path / "grid.nc"
    _write_icon_grid(gridfile)
    ds = xr.Dataset({"field": ("ncells", [0.0, 1.0, 2.0, 3.0])})
    grid = ds.icon.add_grid(gridfile)
    grid.close()

    regular = ds["field"].icon.regrid([0.0, 1.0, 2.0, 3.0], [0.0, 1.0], method="nearest", ltranslon=False)
    assert regular.dims == ("Latitude", "Longitude")
    assert regular.sel(Longitude=1.0, Latitude=0.0).item() == pytest.approx(1.0)

    triangles, colors, _ = ds["field"].icon.tri_data(ltranslon=False)
    assert len(triangles) == 4
    assert colors.shape[0] == 4

    fig, ax = plt.subplots()
    ds["field"].icon.tri_plot(ax, ltranslon=False, add_colorbar=False)
    assert any(isinstance(collection, PolyCollection) for collection in ax.collections)
    plt.close(fig)

    fig, ax = plt.subplots()
    ds["field"].icon.tri_plot(ax, ltranslon=False, add_colorbar=False, projection=None)
    assert any(isinstance(collection, PolyCollection) for collection in ax.collections)
    plt.close(fig)

    derived = (ds["field"] + 1.0).assign_coords(clon=ds["clon"], clat=ds["clat"])
    derived_triangles, derived_colors, _ = derived.icon.tri_data(ltranslon=False)
    assert len(derived_triangles) == 4
    assert derived_colors.shape[0] == 4

    derived_without_centers = (ds["field"] + 1.0).reset_coords(["clon", "clat"], drop=True)
    fallback_triangles, fallback_colors, _ = derived_without_centers.icon.tri_data(ltranslon=False)
    assert len(fallback_triangles) == 4
    assert fallback_colors.shape[0] == 4

    subset = derived_without_centers.isel(ncells=[1, 3])
    subset_triangles, subset_colors, _ = subset.icon.tri_data(ltranslon=False)
    assert len(subset_triangles) == 2
    assert subset_colors.shape[0] == 2

    fig, ax = plt.subplots()
    subset.viz.tricontourf(ax, backend="polycollection", add_colorbar=False)
    assert any(isinstance(collection, PolyCollection) for collection in ax.collections)
    plt.close(fig)


def test_art_dataarray_plume_diagnostics():
    da = xr.DataArray(
        [[0.0, 2.0], [3.0, 0.0], [5.0, 1.0]],
        dims=("height", "ncells"),
    )
    z_mc = xr.DataArray(
        [[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]],
        dims=("height", "ncells"),
    )
    rho = xr.DataArray(np.ones((3, 2)), dims=("height", "ncells"))
    dz = xr.DataArray([10.0, 20.0, 30.0], dims=("height",))

    plume = da.art.select_plume(1.0)
    np.testing.assert_allclose(plume.values, [[0.0, 2.0], [3.0, 0.0], [5.0, 0.0]])
    np.testing.assert_allclose(da.art.tracer_load(rho, dz).values, [210.0, 50.0])
    np.testing.assert_allclose(plume.art.plume_top(z_mc).values, [300.0, 100.0])
    np.testing.assert_allclose(plume.art.plume_bottom(z_mc).values, [200.0, 100.0])
    np.testing.assert_allclose(plume.art.max_conc_height(z_mc).values, [300.0, 100.0])
