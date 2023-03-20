import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


@xr.register_dataarray_accessor('viz')
class PlotAccessor(object):
    def __init__(self, da):
        self._obj = da
    
    def plot(self):
        print('Wait: Figures are on the way.')
        pass