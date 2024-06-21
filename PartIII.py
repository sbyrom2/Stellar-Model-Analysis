# This code was developed by referencing notebooks written by J. Tayar and Z. Claytor
# particularly the follow notebook written by J. Tayar: https://github.com/jtayar/kiauhoku_stuff/blob/main/AIPheGI6.ipynb
import numpy as np
import matplotlib.pyplot as plt
import kiauhoku as kh
import numba.np.unsafe.ndarray 
import pandas as pd

'''
The {model name}_iso(age, metallicity, masses) functions accept an age, metallicity, and a list of masses.
At a certain mass limit, the star will fit well to the grid. This limit gets lower at high ages
and changes based on the metallicty as well. Limits for near solar metallicity are listed, but I 
recommend checking high masses to determine the exact limit for your input age and metallicity.
For example, I ran masses = np.linspace(1.2,1.35,100) with verbose=True to determine the upper mass
limit for old stars (4-6 Gyr) near solar metallicity.

The typical mass list is np.linspace(.63, 1.3, 1000)
This takes a while to run.

'''
def yrec_iso(age, metallicity, masses, verbose=True):
    #for age=4, mass limit is ~1.3766
    #for age=5, mass limit is ~1.289
    #for age=6, mass limit is ~1.2267
    gridnames = []
    models = []
    for i in range(len(masses)):
        if verbose:
            print(masses[i])
        star={'age':age, 'met':metallicity, 'mass':masses[i] }
        model1=fit_yrec(star,scale=(0.1, 0.1, 0.1), tol=1e-6, verbose=verbose)
        models.append(model1)
    models = pd.concat(models, axis=1)
    models.columns = masses
    return models

def mist_iso(age, metallicity, masses, verbose=True):
    #for age=4, mass limit is ~1.349
    #for age=5, mass limit is ~1.2646
    #for age=6, mass limit is ~1.199
    gridnames = []
    models = []
    for i in range(len(masses)):
        if verbose:
            print(masses[i])
        star={'age':age, 'met':metallicity, 'mass':masses[i] }
        model1=fit_mist(star,scale=(0.1, 0.1, 0.1), tol=1e-6, verbose=verbose)
        models.append(model1)
    models = pd.concat(models, axis=1)
    models.columns = masses
    return models

def dart_iso(age, metallicity, masses, verbose=True):
    #for age=4, masses: (0.63, 1.37, 500)
    #for age=5, masses: (0.63, 1.2791, 500)
    #for age=6, masses: (0.63, 1.2144, 500)
    gridnames = []
    models = []
    for i in range(len(masses)):
        if verbose:
            print(masses[i])
        star={'age':age, 'met':metallicity, 'mass':masses[i] }
        model1=fit_dart(star,scale=(0.1, 0.1, 0.1), tol=1e-6, verbose=verbose)
        models.append(model1)
    models = pd.concat(models, axis=1)
    models.columns = masses
    return models

def gars_iso(age, metallicity, masses, verbose=True):
    #for age=4, mass limit is ~1.3717
    #for age=5, mass limit is ~1.2832
    #for age=6, mass limit is ~1.2172
    gridnames = []
    models = []
    for i in range(len(masses)):
        if verbose:
            print(masses[i])
        star={'age':age, 'met':metallicity, 'mass':masses[i] }
        model1=fit_gars(star,scale=(0.1, 0.1, 0.1), tol=1e-6, verbose=verbose)
        models.append(model1)
    models = pd.concat(models, axis=1)
    models.columns = masses
    return models


'''The fit_{model grid} functions accept a star and fit it to the specified model grid
Credit given to J. Tayar for writing these functions
'''
def fit_yrec(star, *args, **kwargs):
    gridnames = []
    models = []
    for gname, interp in zip(
        ['yrec'],
        [yrec]):
        model, fit = interp.gridsearch_fit(star, *args, **kwargs)
        if fit.success:
            gridnames.append(gname)
            models.append(
                model[['initial_mass', 'initial_met', 'eep', 'mass', 'teff', 'lum', 'met', 'logg', 'age']]
            )
    models = pd.concat(models, axis=1)
    models.columns = gridnames
    models
    return models

def fit_gars(star, *args, **kwargs):
    gridnames = []
    models = []
    for gname, interp in zip(
        ['gars'],
        [gars]):
        model, fit = interp.gridsearch_fit(star, *args, **kwargs)
        if fit.success:
            gridnames.append(gname)
            models.append(
                model[['initial_mass', 'initial_met', 'eep', 'mass', 'teff', 'lum', 'met', 'logg', 'age']]
            )
    models = pd.concat(models, axis=1)
    models.columns = gridnames
    models
    return models

def fit_mist(star, *args, **kwargs):
    gridnames = []
    models = []
    for gname, interp in zip(
        ['mist'],
        [mist]):
        model, fit = interp.gridsearch_fit(star, *args, **kwargs)
        if fit.success:
            gridnames.append(gname)
            models.append(
                model[['initial_mass', 'initial_met', 'eep', 'mass', 'teff', 'lum', 'met', 'logg', 'age']]
            )
    models = pd.concat(models, axis=1)
    models.columns = gridnames
    models
    return models


def fit_dart(star, *args, **kwargs):
    gridnames = []
    models = []
    for gname, interp in zip(
        ['dart'],
        [dart]):
        model, fit = interp.gridsearch_fit(star, *args, **kwargs)
        if fit.success:
            gridnames.append(gname)
            models.append(
                model[['initial_mass', 'initial_met', 'eep', 'mass', 'teff', 'lum', 'met', 'logg', 'age']]
            )
    models = pd.concat(models, axis=1)
    models.columns = gridnames
    models
    return models





'''GRIDSEARCH SETUP'''
# use grid points between ZAMS (201) and RGBump (605)
qstring = '0.6 <= initial_mass <= 2 and -1.0 <= initial_met <= 0.5 and 201 <= eep <= 605'

# Whether to fit evolved metallicity (True) or use the initial metallicity.
# False is probably fine if you're not on the giant branch.
evolve_met = False

# load grid, remove unwanted rows
yrec = kh.load_eep_grid("yrec").query(qstring)
# set column names to some standard
yrec['mass'] = yrec['Mass(Msun)']
yrec['teff'] = 10**yrec['Log Teff(K)']
yrec['lum'] = 10**yrec['L/Lsun']
if evolve_met:
    yrec['met'] = np.log10(yrec['Zsurf']/yrec['Xsurf']/0.0253)
else:
    yrec['met'] = yrec.index.get_level_values('initial_met')
yrec['age'] = yrec['Age(Gyr)']
# set name for readability of output
yrec.set_name('yrec')
# cast to interpolator
yrec = yrec.to_interpolator()

mist = kh.load_eep_grid("mist").query(qstring)
mist['mass'] = mist['star_mass']
mist['teff'] = 10**mist['log_Teff']
mist['lum'] = 10**mist['log_L']
if evolve_met:
    mist['met'] = mist['log_surf_z'] - np.log10(mist['surface_h1']*0.0173)
else:
    mist['met'] = mist.index.get_level_values('initial_met')
mist['logg'] = mist['log_g']
mist['age'] = mist['star_age'] / 1e9
mist.set_name('mist')
mist = mist.to_interpolator()

dart = kh.load_eep_grid("dartmouth").query(qstring)
dart['mass'] = dart.index.to_frame()['initial_mass']
dart['teff'] = 10**dart['Log T']
dart['lum'] = 10**dart['Log L']
if evolve_met:
    dart['met'] = np.log10(dart['(Z/X)_surf']/0.0229)
else:
    dart['met'] = dart.index.get_level_values('initial_met')
dart['logg'] = dart['Log g']
dart['age'] = dart['Age (yrs)'] / 1e9
dart.set_name('dart')
dart = dart.to_interpolator()

gars = kh.load_eep_grid("garstec").query(qstring)
gars['mass'] = gars['M/Msun']
gars['teff'] = gars['Teff']
gars['lum'] = 10**gars['Log L/Lsun']
if evolve_met:
    gars['met'] = np.log10(gars['Zsurf']/gars['Xsurf']/0.0245)
else:
    gars['met'] = gars.index.get_level_values('initial_met')
gars['age'] = gars['Age(Myr)'] / 1e3
gars.set_name('gars')
gars = gars.to_interpolator()