### Classes for calculating reflecteance for different implementations of the Hapke slab models. ###
# To Do: add other slab model variants

import numpy as np
from typing import List, Union
import astropy
from astropy import units as u
import miepython as mie
from scipy import interpolate
import scipy.integrate as integrate

from .materials import OpticalConstant, Material
from .base_model import HapkeModel

class SlabModel(HapkeModel):
    """ Basic Slab Model."""

    def _Se(self, n,k):
        """ Calculate Se, the total fraction of light externally incident on surface that is specularly reflected
        -- Hapke equation 5.36 for a given n and k (float values)."""
        def F1(theta): #G1 in hapke - eq 4.48
            return np.sqrt(.5*((n*n-k*k-np.sin(theta)**2)+np.sqrt((n*n-k*k-np.sin(theta)**2)**2+4*n*n*k*k)))
        def F2(theta): #G2 in hapke - eq 4.49
            return np.sqrt(.5*(-(n*n-k*k-np.sin(theta)**2)+np.sqrt((n*n-k*k-np.sin(theta)**2)**2+4*n*n*k*k)))
        def Rperp(theta): # eq 4.52
            return ((np.cos(theta)-F1(theta))**2+F2(theta)**2)/((np.cos(theta)+F1(theta))**2+F2(theta)**2)
        def Rpar(theta): # eq 4.53
            r1=((n**2-k**2)*np.cos(theta)-F1(theta))**2+(2*n*k*np.cos(theta)-F2(theta))**2
            r2=((n**2-k**2)*np.cos(theta)+F1(theta))**2+(2*n*k*np.cos(theta)+F2(theta))**2
            return r1/r2
        def re(theta): # eq 5.36
            return ((Rperp(theta)+Rpar(theta))*np.cos(theta)*np.sin(theta)) 
                
        s = integrate.quad(re, 0, np.pi/2) # integrate eq 5.36 from 0 to pi/2
        return s[0]
    
    def get_Se(self, n,k, waves):
        """ Calculate Se"""
        assert len(n)==len(k)==len(waves)
        return np.array([self._Se(n[i], k[i]) for i in range(len(waves))])
    
    def get_Si(self, n,k, waves):
        """ Calculate Si, the internal scattering coefficient -- eq 6.21 for a given n and k (float values).
            Si is same as Se (eq 5.36) but with 1/m instead of m (where m=n(1-jk)) -- inverse of a complex number!!!"""
        assert len(n)==len(k)==len(waves)
        return np.array([self._Se(n[i]/(n[i]**2+k[i]**2), -k[i]/(n[i]**2+k[i]**2)) for i in range(len(waves))])
    
    def get_single_scattering_albedo(self, temp, waves, grainsz, params) -> np.ndarray:
        """ Calculate the single scattering albedo for Hapke slab model.

            n,k: 1darray: real and imaginary parts of the refractive index 
            waves: 1darray: wavelengths, must be in microns
            grainsz: float: grain size in microns
            ssa_params: dict: {'s': float (default=1e-16): default is very little scattering, }
        """
        if params is None:
            params = {'scattering_coeff': 1e-16} #very little scattering
        
        n,k = self.get_nk(temp, waves=waves) # get n and k for the given temperature
        # get Se, Si by directly integrating eq 5.36
        Se = self.get_Se(n,k, waves)
        Si = self.get_Si(n,k,waves)
        
        # hapke equation ??? (23 in paper) -- slab approximation
        alpha = 4*np.pi*k/waves
        s = params['scattering_coeff']
    
        ri=(1-np.sqrt(alpha/(alpha+s)))/(1+np.sqrt(alpha/(alpha+s))) # bihemispherical reflectance
        Theta=(ri+np.exp(-np.sqrt(alpha*(alpha+s))*grainsz))/(1+ri*np.exp(-np.sqrt(alpha*(alpha+s))*grainsz))
        w = Se+(1-Se)*(1-Si)*Theta/(1-Si*Theta)  # eqn ??
        return w