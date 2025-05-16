import numpy as np
from dataclasses import dataclass
from typing import List, Union
import astropy
from astropy import units as u
import miepython as mie
from scipy import interpolate
import scipy.integrate as integrate


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

        # Useful values
        self.R = ((n-1)**2 + k**2)/((n+1)**2 + k**2) # Fresnell coefficient
        self.alpha = 4*np.pi*self.k/self.wv_microns # Absorptoin coefficient


    def calculate_reflection(self, theta):
        """ Calculate reflection values (Rperp, Rparallel) for a given incidence angle theta."""
        # factors and specular reflectance constants (eqs. 4.44-4.49)
        n,k = self.n, self.k
        F1=np.sqrt(.5*((n*n-k*k-np.sin(theta)**2)+np.sqrt((n*n-k*k-np.sin(theta)**2)**2+4*n*n*k*k))) #
        F2=np.sqrt(.5*(-(n*n-k*k-np.sin(theta)**2)+np.sqrt((n*n-k*k-np.sin(theta)**2)**2+4*n*n*k*k)))
    
        Rperp=((np.cos(theta)-F1)**2+F2**2)/((np.cos(theta)+F1)**2+F2**2)
           
        r1=((n**2-k**2)*np.cos(theta)-F1)**2+(2*n*k*np.cos(theta)-F2)**2
        r2=((n**2-k**2)*np.cos(theta)+F1)**2+(2*n*k*np.cos(theta)+F2)**2
        Rparallel = r1/r2

        return Rperp, Rparallel

    def plot_nk(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,1, sharex=True)
        ax[0].plot(self.wv, self.n, label='n')
        ax[0].plot(self.wv, self.k, label='k')
        ax[0].legend()

        ax[1].plot(self.wv, self.R, label='R')
        ax[1].legend()
        ax[1].set_xlabel('Wavelength (um)')

        ax[0].set_title(f"{self.name} at {self.temp}K")

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
    
class HapkeModel:
    """ Class for performing Hapke modeling on a given material."""
    def __init__(self, material: Material):
        self.material = material
        self.name = material.name

    def get_single_scattering_albedo(self, n, k, waves, grainsz, params=None) -> np.ndarray:
        """ Calculate the single scattering albedo for a single particle given a temperature and grain size. 
            Note: this is the single scattering albedo, not the reflectance. The reflectance is calculated using the HapkeModel.calculate_reflectance.

            n,k: 1darray: real and imaginary parts of the refractive index 
            waves: 1darray: wavelengths, must be in microns
            grainsz: float: grain size in microns
            params: dict: parameters for the single scattering albedo calculation. None needed for single particle model.
        """

        # get scattering and extinction coefficients miepython
        x = np.pi*grainsz / waves # size parameter, Hapke 5.8
        m =  fn(waves)-1.0j*fk(waves)# complex refractive index
        qext, qsca, qback, g = mie.efficiencies_mx(m, x)
        w=qsca/qext
        return w
    
    def calculate_reflectance(self, temp: float, grainsz: Union[float, astropy.units.Quantity], nearesttemp=True, waves: Union[np.ndarray, astropy.units.Quantity] = None, ssa_params:dict = None) -> np.ndarray):
        """ Calculate the reflectance spectrum for a given temperature and grain size. 
            temp: float: temperature in K
            grainsz: float or astropy quantity: grain size diameter -- must be in microns if float provided
            nearesttemp: bool: if True, will use the nearest temperature if exact match is not found
            waves: 1darray or astropy.units.Quantity array: wavelengths to calculate reflectance for -- if None, will use the wavelengths from the optical constant, if not units provided must be in microns
            ssa_params: dict: parameters for the single scattering albedo calculation. If None, will use default values.
        """
        nk = self.material.get_nk(temp, nearest=nearesttemp)
        fn = interpolate.interp1d(nk.wv_microns, nk.n) # interpolation functions for n and k
        fk = interpolate.interp1d(nk.wv_microns, nk.k)
        grainsz = grainsz.to(u.micron).value if isinstance(grainsz, astropy.units.Quantity) else grainsz
        if isinstance(waves, astropy.units.Quantity):
            waves = waves.to(u.micron).value
        elif isinstance(waves, np.ndarray):
            waves = waves
        elif waves is None:
            waves = nk.wv_microns

        # get single scattering albedo using scattering and extinction coefficients from miepython
        w = self.get_single_scattering_albedo(fn(waves), fk(waves), waves, params=ssa_params)
        
        # convert single scattering albedo to reflectance (physical albedo eq 11.32)
        lam=np.sqrt(1-w)
        r0=(1-lam)/(1+lam)
        alb=0.49*r0 + 0.196*r0**2 # using approximation (eq 11.33) for now.
        #To Do: actually solve chandrasekhar equations here and solve eq 11.32!
        return alb
            




#class SingleGrainModel(HapkeModel):




class SlabModel(HapkeModel):

    def _Se(n,k):
        """ Calculate Se, the total fraction of light externally incident on surface that is specularly reflected
        -- Hapke equation 5.36 for a given n and k (float values)."""
        def F1(theta): #G1 in hapke - 4.48
            return np.sqrt(.5*((n*n-k*k-np.sin(theta)**2)+np.sqrt((n*n-k*k-np.sin(theta)**2)**2+4*n*n*k*k)))
        def F2(theta): #G2 in hapke - 4.49
            return np.sqrt(.5*(-(n*n-k*k-np.sin(theta)**2)+np.sqrt((n*n-k*k-np.sin(theta)**2)**2+4*n*n*k*k)))
        def Rperp(theta): # eq 4.52
            return ((np.cos(theta)-F1(theta))**2+F2(theta)**2)/((np.cos(theta)+F1(theta))**2+F2(theta)**2)
        def Rpar(theta): # eq 4.53
            r1=((n**2-k**2)*np.cos(theta)-F1(theta))**2+(2*n*k*np.cos(theta)-F2(theta))**2
            r2=((n**2-k**2)*np.cos(theta)+F1(theta))**2+(2*n*k*np.cos(theta)+F2(theta))**2
            return r1/r2
        def re(theta): 
            return ((Rperp(theta)+Rpar(theta))*np.cos(theta)*np.sin(theta)) # eq 5.36
                
        s = integrate.quad(re, 0, np.pi/2) # integrate eq 5.36 from 0 to pi/2
        return s[0]
    
    def get_Si(n,k, waves):
        """ Calculate Se"""
        return np.array([self._Se(n[i], k[i]) for wv in range(len(waves))])
    
    def get_Si(n,k, waves):
        """ Calculate Si, the internal scattering coefficient -- eq 6.21 for a given n and k (float values).
            Si is same as 5.31 but with 1/m instead of m (where m=n(1-jk)) -- inverse of a complex number!!!"""
        return np.array([self._Se(n[i]/(n[i]**2+k[i]**2), -k[i]/(n[i]**2+k[i]**2)) for wv in range(len(waves))])
    
    def get_single_scattering_albedo(self, n, k, waves, grainsz, params) -> np.ndarray:
        """ Calculate the single scattering albedo for Hapke slab model.

            n,k: 1darray: real and imaginary parts of the refractive index 
            waves: 1darray: wavelengths, must be in microns
            grainsz: float: grain size in microns
            ssa_params: dict: {'s': float (default=1e-16): default is very little scattering, }
        """
        if ssa_params is None:
            ssa_params = {'scattering_coeff': 1e-16} #very little scattering
        
        # get Se, Si by directly integrating eq 5.36
        Se = self.get_Se(n,k, waves)
        Si = self.get_Si(n,k,waves)
        
        # hapke equation ??? (23 in paper) -- slab approximation
        alpha = 4*np.pi*k/waves
        s = ssa_params['scattering_coeff']
    
        ri=(1-np.sqrt(alpha/(alpha+s)))/(1+np.sqrt(alpha/(alpha+s))) # bihemispherical reflectance
        Theta=(ri+np.exp(-np.sqrt(alpha*(alpha+s))*grainsz))/(1+ri*np.exp(-np.sqrt(alpha*(alpha+s))*grainsz))
        w = Se+(1-Se)*(1-Si)*Theta/(1-Si*Theta)  # eqn ??
        return w
        

        