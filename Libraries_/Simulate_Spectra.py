# -*- coding: utf-8 -*-

"""
Created on Thu Aug 12 12:40:50 2021

@author: Bernardo Arturo Salcido Santacruz 
"""
##################################
import numpy as np
import matplotlib.pyplot as plt

from rdkit.Chem import Draw
from matplotlib.offsetbox import AnnotationBbox, OffsetImage 


def lineshape(nu,gamma0,nu0,f,a = 0,A = 1):
    gamma_nu = 2*gamma0/(1+np.exp(a*(nu -nu0))) 

    G_nu = (A/gamma_nu) * np.sqrt(4*np.log(2)/np.pi) * np.exp(-4 *np.log(2)*(((nu -nu0)/gamma_nu)**2))
    L_nu = (2*A/np.pi*gamma_nu)/ (1+4*(((nu -nu0)/gamma_nu)**2)) 

    y_nu = (f*L_nu) + ((1-f)*G_nu)    
    
    return y_nu

def SimulateSpectra(w_abs,w_em,FWHM_abs,FWHM_em,w_min = False, w_max=False,mol= False,solvent = False, Save = False):
    
    wavelengt = np.linspace(200,1000,1000)
    
    
    a = 0.5/(FWHM_abs*2)
    
    f = 0.5 #
    ### Absortion
    I_abs = lineshape(wavelengt,FWHM_abs,w_abs,f,a) 
    I_abs = I_abs /np.max(I_abs) ## Normalize 
    ### Emission
    
    a = 0.5/(FWHM_em*2)
    I_em = lineshape(wavelengt,FWHM_em,w_em,f,-a) 
    I_em = I_em /np.max(I_em) ## Normalize 
    ### Figure
    fig, ax = plt.subplots(dpi = 300)
    ax.plot(wavelengt, I_abs,label='Absortion  ('+str(round(w_abs))+' nm)')
    ax.plot(wavelengt, I_em,label='Emission  ('+str(round(w_em))+' nm)')
    
    if w_max != False:
        
        ax.set_xlim((w_min,w_max))
    
    if mol != False:
        
        Draw.MolToFile(mol.mol2D,'TEMP/image_mol.png')  
        
        
        Molecule = plt.imread('TEMP/image_mol.png')
        
        
        imagebox = OffsetImage(Molecule,zoom=0.4)
        imagebox.image.axes = ax
        
        XY = [w_em+(FWHM_em*2.2),0.65]
        add_Molecule = AnnotationBbox(imagebox,XY,xybox = (30.,-30.),frameon =False,xycoords='data',boxcoords = "offset points",pad=0.5)
        ax.add_artist(add_Molecule)
    

    
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Intencity [a.u.]')
    plt.title('Absoriton an emission spectra in '+str(solvent))
    ax.legend()
    
        
    
    plt.show()

#  Wavelengt to Energy Electronvolt
#--------------------------------
def nm_to_m(nm):
    m = (10**(-9))*nm
    return m 

def m_to_nm(m):
    nm = (10**(+9))*m
    return nm 

#--------------------------------
def nm_to_cm(nm):
    cm = (10**(-7))*nm
    return cm 

def cm_to_nm(cm):
    nm = (10**(7))*cm
    return nm 

#--------------------------------
def wavelengt_to_wavenumber(wavelengt):
    # wavelengt [nm] to frequency [cm-1]
    wavenumber = 1/ nm_to_cm(wavelengt)
    
    return wavenumber

def wavenumber_to_wavelengt(wavenumber):
    # wavelengt [nm],  frequency [cm-1]
    wavelengt = cm_to_nm(1/wavenumber)
    
    return wavelengt

#--------------------------------
def wavelengt_to_energy(wavelengt):
    # wavelengt [nm] to energy [eV]
    
    c = 299792458 #[m/s]
    h = 4.1355667696 *10**(-15) #[eV/s]
    
    energy = (h*c)/nm_to_m(wavelengt)
    
    return energy

def energy_to_wavelengt(energy):
    # wavelengt [nm] to energy [eV]
    
    c = 299792458 #[m/s]
    h = 4.1355667696 *10**(-15) #[eV/s]
    
    wavelengt = m_to_nm((h*c)/energy)
    
    return wavelengt
#%%


