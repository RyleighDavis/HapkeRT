### Classes for keeping track of temperature dependent optical constants for a given material ###

import numpy as np
from dataclasses import dataclass
from typing import List, Union
import astropy
from astropy import units as u
from scipy import interpolate

@dataclass
class OpticalConstant:
    """ Class for keeping track of a material properties at a specific temperature.
        wavelength: 1d array of wavelengths
        n: 1d arrays of real optical constant
        k: 1d array of imaginary optical constant
        temp: float (K)
        wavelength_units: astropy unit: default is microns
        source: str (optional): ßßßsource of data for easy reference
    """
    
    wv: np.ndarray # wavelengths 
    n: np.ndarray
    k: np.ndarray
    temp: np.ndarray # temperature in K
    wv_units: astropy.units.core.Unit = u.micron
    source: str = 'unknown'
    name: str = ''

    def __init__(self, wv: np.ndarray, n: np.ndarray, k: np.ndarray, temp: int, wv_units: astropy.units.core.Unit, source: str = 'unknown', name: str = ''):
        self.source = source
        self.wv = wv
        self.wv_microns = (self.wv * wv_units).to(u.micron).value
        self.n = n
        if np.iscomplex(k).all():
            self.k= np.imag(k)
        else:
            self.k = k
        self.temp = temp
        self.wv_units = wv_units
        self.source = source
        self.name = name

        # interpolation function for n and k
        self.fn = interpolate.interp1d(self.wv_microns,n)
        self.fk = interpolate.interp1d(self.wv_microns,k)

        # Useful values
        self.R = ((n-1)**2 + k**2)/((n+1)**2 + k**2) # Fresnell coefficient
        self.alpha = 4*np.pi*self.k/self.wv_microns # Absorptoin coefficient


    # def calculate_reflection(self, theta):
    #     """ Calculate reflection values (Rperp, Rparallel) for a given incidence angle theta."""
    #     # factors and specular reflectance constants (eqs. 4.44-4.49)
    #     n,k = self.n, self.k
    #     F1=np.sqrt(.5*((n*n-k*k-np.sin(theta)**2)+np.sqrt((n*n-k*k-np.sin(theta)**2)**2+4*n*n*k*k))) #
    #     F2=np.sqrt(.5*(-(n*n-k*k-np.sin(theta)**2)+np.sqrt((n*n-k*k-np.sin(theta)**2)**2+4*n*n*k*k)))
    
    #     Rperp=((np.cos(theta)-F1)**2+F2**2)/((np.cos(theta)+F1)**2+F2**2)
           
    #     r1=((n**2-k**2)*np.cos(theta)-F1)**2+(2*n*k*np.cos(theta)-F2)**2
    #     r2=((n**2-k**2)*np.cos(theta)+F1)**2+(2*n*k*np.cos(theta)+F2)**2
    #     Rparallel = r1/r2

    #     return Rperp, Rparallel

    def interpolate_nk(self, wv: np.ndarray) -> np.ndarray:
        """ Interpolate n and k for a given wavelength array. """
        if isinstance(wv, astropy.units.Quantity):
            wv = wv.to(u.micron).value
        elif isinstance(wv, np.ndarray):
            wv = wv
        else:
            raise ValueError("Wavelength must be a numpy array (implied units of microns) or astropy quantity.")
        
        n = self.fn(wv)
        k = self.fk(wv)
        return n, k

    def plot_nk(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.wv, self.n, label='n')
        ax.plot(self.wv, self.k, label='k')
        ax.legend()
        ax.set_xlabel('Wavelength (um)')
        ax.set_title(f"{self.name} at {self.temp}K")

        # ax[1].plot(self.wv, self.R, label='R')
        # ax[1].legend()
        

@dataclass
class Material:
    """ Class for keeping track of all temperature dependent optical constants for a given material. """
    name: str
    nk: List[OpticalConstant]


    def __init__(self, name: str, nk: List[OpticalConstant]):
        self.name = name
        self.nk = nk

    def get_nk(self, temp:float, nearest: bool = True) -> OpticalConstant:
        """ Get the optical constant for a given temperature. If an exact match is not found,
         either return nothing (if nearest=False) or the nearest tempeature (if nearest=True). """
        # To Do: possibly add functionality to interpolate between two temperatures
        found = False
        for opt_const in self.nk:
            if opt_const.temp == temp:
                found = True
                return opt_const
        if not found and nearest:
            # Find the nearest temperature
            temp_diffs = [abs(opt_const.temp - temp) for opt_const in self.nk]
            nearest_index = np.argmin(temp_diffs)
            print(f"Did not find exact match for {self.name} at {temp}K. Using nearest temperature {self.nk[nearest_index].temp}K.")
            return self.nk[nearest_index]
        raise ValueError(f"No optical constant found for {self.name} at {temp}K. To find nearest temperature, set nearest=True.")