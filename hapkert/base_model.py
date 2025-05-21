### Basic model class for Hapke modelling. Converts n and k to single scattering albedo and the
### single scattering albedo to reflectance and other types of albedo.

import numpy as np
from typing import Union
import astropy
from astropy import units as u
import miepython as mie
from scipy import interpolate

from .materials import Material

class HapkeModel:
    """ Class for performing Hapke modeling on a given material."""
    def __init__(self, material: Material):
        self.material = material
        self.name = material.name

    def get_nk(self, temp: float, nearest: bool = True, waves=None):
        """ Get the optical constant for a given temperature. 
            temp: float: temperature in K
            nearest: bool: if True, will use the nearest temperature if exact match is not found
            waves: 1darray: wavelengths to calculate reflectance for -- if None, will use the wavelengths 
                   from original optical constant measurement, if not units provided must be in microns
        """
        nk = self.material.get_nk(temp, nearest=nearest)
        if waves is None:
            waves = nk.wv_microns
        return nk.interpolate_nk(waves)

    def get_single_scattering_albedo(self, temp, waves, grainsz, params=None) -> np.ndarray:
        """ Calculate the single scattering albedo for a single particle given a temperature and grain size. 
            Note: this is the single scattering albedo, not the reflectance. The reflectance is calculated using the HapkeModel.calculate_reflectance.

            temp: float: temperature in K to pull n and k from
            waves: 1darray: wavelengths, must be in microns
            grainsz: float: grain size in microns
            params: dict: parameters for the single scattering albedo calculation. None needed for single particle model.
        """

        # get scattering and extinction coefficients miepython
        x = np.pi*grainsz / waves # size parameter, Hapke 5.8
        n,k = self.get_nk(temp, waves=waves) # get n and k for the given temperature
        m =  n-1.0j*k# complex refractive index
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        w=qsca/qext
        return w
    
    def calculate_reflectance(self, temp: float, grainsz: Union[float, astropy.units.Quantity], nearesttemp=True, waves: Union[np.ndarray, astropy.units.Quantity] = None, ssa_params:dict = None) -> np.ndarray:
        """ Calculate the reflectance spectrum for a given temperature and grain size. 
            temp: float: temperature in K
            grainsz: float or astropy quantity: grain size diameter -- must be in microns if float provided
            nearesttemp: bool: if True, will use the nearest temperature if exact match is not found
            waves: 1darray or astropy.units.Quantity array: wavelengths to calculate reflectance for -- if None, will use the wavelengths from the optical constant, if not units provided must be in microns
            ssa_params: dict: parameters for the single scattering albedo calculation. If None, will use default values.
        """
        # nk = self.material.get_nk(temp, nearest=nearesttemp)
        # fn = interpolate.interp1d(nk.wv_microns, nk.n) # interpolation functions for n and k
        # fk = interpolate.interp1d(nk.wv_microns, nk.k)
        grainsz = grainsz.to(u.micron).value if isinstance(grainsz, astropy.units.Quantity) else grainsz
        if isinstance(waves, astropy.units.Quantity):
            waves = waves.to(u.micron).value
        elif waves is None:
            waves = self.material.get_nk(temp).wv_microns

        # get single scattering albedo using scattering and extinction coefficients from miepython
        w = self.get_single_scattering_albedo(temp, waves, grainsz, params=ssa_params)
        
        # convert single scattering albedo to reflectance (physical albedo eq 11.32)
        lam=np.sqrt(1-w)
        r0=(1-lam)/(1+lam)
        alb=0.49*r0 + 0.196*r0**2 # using approximation (eq 11.33) for now.
        #To Do: actually solve chandrasekhar equations here and solve eq 11.32!
        return alb