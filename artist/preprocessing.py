import icontoolbox as ic
import os
import xarray as xr

"""
Wrapper for icontoolbox submodule
Author: Cornelius Tai


"""


def IFS4ICON(
    init_date,
    remap_nml=None,
    output_grid=None,
    output_path=None,
    config_yaml=None,
    iconremap_bin="/home/b/b382290/vol_work/icon_tools/dwd_icon_tools/icontools/iconremap",
):
    """
    Wrapper function of icontoolbox.IFS4ICON(), which is shipped with this package as submodule. It is a 1 stop solution to prepare for ICON initialisation using IFS data (initicon_nml::init_mode = 2).
    This function is designed to be used on levante, where a subset of ERA5 renalysis is stored. 3 additional variables are downloaded via the cdsapi, see: https://cds.climate.copernicus.eu/how-to-api for detailed setup of the cdsapi.
    Requirements:
        - cdsapi
        - cdo (Climate Data Operator)

    Input:
        init_date        (str)           : date of the reanalysis, format YYYY-MM-DD
        remap_nml        (list, str)     : List of paths to Namelist files for iconremap tool, format [NAMELIST_REMAP, NAMELIST_INPUT_FIELD], default: None -> namelist will be constructed based on parameters passed
        output_grid      (str)           : path to destination ICON grid file
        output_path      (str)           : path to the directory where the output files will be stored, default: None -> os.getcwd()
        config_yaml      (str)           : path to the configuration YAML file for Namelist construction, default: None -> use default configuration
        iconremap_bin    (str)           : path to the iconremap binary, default: /home/b/b382290/vol_work/icon_tools/dwd_icon_tools/icontools/iconremap
    """

    return ic.IFS4ICON(
        init_date=init_date,
        remap_nml=remap_nml,
        output_grid=output_grid,
        output_path=output_path,
        config_yaml=config_yaml,
        iconremap_bin=iconremap_bin,
    )

