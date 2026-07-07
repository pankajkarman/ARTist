import sys
import types

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import artist
import artist.oem as oem


SPECIES = ["CH4", "CO2"]


def _install_fake_emiproc(monkeypatch, calls):
    monkeypatch.setitem(sys.modules, "emiproc", types.ModuleType("emiproc"))
    module_names = [
        "emiproc.inventories",
        "emiproc.inventories.edgar",
        "emiproc.inventories.edgar.temporal",
        "emiproc.inventories.utils",
        "emiproc.grids",
        "emiproc.regrid",
        "emiproc.profiles",
        "emiproc.profiles.vertical_profiles",
        "emiproc.exports",
        "emiproc.exports.utils",
        "emiproc.exports.icon",
        "emiproc.plots",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class FakeInventory:
        def __init__(self, files=None, use_short_category_names=True, year=None):
            self.files = files
            self.year = year
            self.substances = list(SPECIES)

        def set_profiles(self, profiles, indexes):
            calls.append(("set_profiles", profiles, indexes))

        def set_profile(self, profile, substance=None):
            calls.append(("set_profile", substance, profile.height.tolist()))

        def to_crs(self, crs):
            calls.append(("to_crs", crs))

    class FakeICONGrid:
        def __init__(self, path):
            self.path = path
            self.name = "tiny_icon_grid"
            calls.append(("ICONGrid", str(path)))

    class FakeVerticalProfile:
        def __init__(self, factors, height=None):
            self.factors = np.asarray(factors)
            self.height = np.asarray(height)

    class TemporalProfilesInterpolated:
        THREE_CYCLES = "three_cycles_interpolated"

    class TemporalProfilesTypes:
        THREE_CYCLES = "three_cycles_export"

    def download_edgar_files(local_dir, year=None, substances=None):
        calls.append(("download", str(local_dir), year, list(substances)))
        local_dir.mkdir(parents=True, exist_ok=True)
        for substance in substances:
            (local_dir / "EDGAR_{}_{}.nc".format(year, substance)).write_text("fake")

    def export_icon_oem(inv=None, icon_grid_file=None, output_dir=None, temporal_profiles_type=None):
        calls.append(("export", str(icon_grid_file), str(output_dir), temporal_profiles_type))
        output_dir.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {
                "CH4": ("ncells", [1.0, 2.0]),
                "CO2": ("ncells", [3.0, 4.0]),
            }
        ).to_netcdf(output_dir / "oem_gridded_emissions.nc", engine="scipy")

    sys.modules["emiproc.inventories.edgar"].download_edgar_files = download_edgar_files
    sys.modules["emiproc.inventories.edgar"].EDGARv8 = FakeInventory
    sys.modules["emiproc.inventories.edgar.temporal"].read_edgar_auxilary_profiles = (
        lambda aux, inventory=None: ({"profiles": SPECIES}, {"indexes": [0, 1]})
    )
    sys.modules["emiproc.grids"].ICONGrid = FakeICONGrid
    sys.modules["emiproc.grids"].WGS84_PROJECTED = "WGS84_PROJECTED"
    sys.modules["emiproc.regrid"].remap_inventory = (
        lambda inventory, icon_grid, weights_file=None: calls.append(("remap", icon_grid.name, str(weights_file)))
        or inventory
    )
    sys.modules["emiproc.profiles.vertical_profiles"].VerticalProfile = FakeVerticalProfile
    sys.modules["emiproc.inventories.utils"].country_to_cells = (
        lambda inventory, country_mask_kwargs=None: calls.append(("country_to_cells", country_mask_kwargs))
        or inventory
    )
    sys.modules["emiproc.inventories.utils"].TemporalProfilesInterpolated = TemporalProfilesInterpolated
    sys.modules["emiproc.inventories.utils"].interpolate_temporal_profiles = (
        lambda inventory, output_type=None: calls.append(("interpolate", output_type)) or inventory
    )
    sys.modules["emiproc.exports.utils"].get_temporally_scaled_array = (
        lambda inventory, time_range=None, sum_over_cells=True: calls.append(
            ("temporal_scaled", len(time_range), sum_over_cells)
        )
        or np.ones((len(time_range), len(inventory.substances)))
    )
    sys.modules["emiproc.exports.icon"].TemporalProfilesTypes = TemporalProfilesTypes
    sys.modules["emiproc.exports.icon"].export_icon_oem = export_icon_oem
    sys.modules["emiproc.plots"].plot_inventory = (
        lambda inv, total_only=True, cmap="magma", **kwargs: calls.append(
            ("plot_inventory", inv.year, total_only, cmap, kwargs)
        )
        or plt.gca()
    )


def _write_icon_grid(path):
    xr.Dataset(
        {
            "clon": ("ncells", np.deg2rad([0.5, 1.5])),
            "clat": ("ncells", np.deg2rad([0.5, 0.5])),
            "clon_vertices": (("ncells", "vertices"), np.deg2rad([[0, 1, 0], [1, 2, 1]])),
            "clat_vertices": (("ncells", "vertices"), np.deg2rad([[0, 0, 1], [0, 0, 1]])),
        }
    ).to_netcdf(path, engine="scipy")


def _write_raw_edgar(path, substance, value):
    emissions = xr.DataArray(
        np.full((4, 6), value, dtype=float),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-1.5, 1.5, 4), "lon": np.linspace(10, 15, 6)},
        attrs={"substance": substance, "year": "2022", "units": "Tonnes"},
    )
    xr.Dataset({"emissions": emissions}).to_netcdf(path, engine="scipy")


def test_oem_workflow_with_ch4_co2(monkeypatch, tmp_path):
    calls = []
    _install_fake_emiproc(monkeypatch, calls)

    edgar_dir = tmp_path / "edgar"
    aux_dir = tmp_path / "aux"
    out_dir = tmp_path / "out"
    gridfile = tmp_path / "icon_grid.nc"
    aux_dir.mkdir()
    _write_icon_grid(gridfile)

    ds = xr.Dataset()
    grid = ds.icon.add_grid(gridfile)
    grid.close()
    gridded = ds.oem.map_edgar(
        edgar_directory=edgar_dir,
        year=2022,
        species=SPECIES,
        output_dir=out_dir,
        aux_data_path=aux_dir,
    )

    assert set(gridded.data_vars) == set(SPECIES)
    assert ds.attrs["_artist_oem_species"] == SPECIES
    assert ("download", str(edgar_dir), 2022, SPECIES) in calls
    assert any(call[0] == "export" for call in calls)

    assert ds.oem.plot_raw_edgar().__class__.__name__ == "Axes"
    ax = ds.oem.plot_mapped_emissions(gridded, variable="CH4", add_colorbar=False)
    assert ax.get_xlabel() == "Longitude"
    assert ax.get_ylabel() == "Latitude"
    gridded.close()
    plt.close("all")


def test_plot_raw_edgar_from_netcdf_directory(tmp_path):
    edgar_dir = tmp_path / "edgar"
    edgar_dir.mkdir()
    _write_raw_edgar(edgar_dir / "EDGAR_CH4_2022_AGS.nc", "CH4", 1.0)
    _write_raw_edgar(edgar_dir / "EDGAR_CO2_2022_AGS.nc", "CO2", 2.0)

    axes = oem.plot_raw_edgar(edgar_directory=edgar_dir, year=2022, species=SPECIES, coarsen=1)

    assert isinstance(axes, list)
    assert len(axes) == 2
    assert axes[0].get_title() == "Raw EDGAR CH4 emissions (2022)"
    assert axes[1].get_title() == "Raw EDGAR CO2 emissions (2022)"
    assert len(axes[0].collections) == 1
    plt.close("all")


def test_plot_raw_edgar_requires_species_for_directory(tmp_path):
    with pytest.raises(ValueError, match="species is required"):
        oem.plot_raw_edgar(edgar_directory=tmp_path)
