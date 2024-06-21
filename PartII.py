import numpy as np
import numba.np.unsafe.ndarray
import kiauhoku as kh
from astropy.table import Table, join
import os.path

'''
List of the APOGEE_IDs for stars that cause the kernel to crash
This has been recorded for M67 and NGC 188
If you work with a new cluster, run get_model_stars with verbose=True and add the stars that cause crashes to this dictionary
'''
skip={'yrec':['2M08510483+1145568', '2M08511428+1208493', '2M08505306+1131201', '2M08511710+1148160', '2M08494102+1138105', #M67 (5)
              '2M01045854+8531503', '2M00375455+8430418', '2M00281873+8455415'], #NGH 188
      'mist':[],
      'dart':['2M08510483+1145568', '2M08511428+1208493', '2M08505306+1131201', '2M08501301+1150224', #M67 (4)
              '2M03334146+4443428','2M03292552+4550338', '2M03295472+4546525', '2M03310220+4526009', '2M00405269+8548333', 
             '2M00281873+8455415'], # NGC 188
      'gars':['2M08510922+1153141', '2M08511428+1208493', '2M08505306+1131201', #M67 (3)
              '2M03325370+4534004', '2M03325151+4554485', '2M03312658+4549449', '2M00490560+8526077', '2M01085160+8517561',
              '2M00350924+8517169', '2M00405169+8415022', '2M00320079+8511465']} #NGC 188

clusters_stars=Table.read("occam_dr17_stars.fits", format="fits")
clusters_stars.add_index("CLUSTER")
clusters_stars.add_index("APOGEE_ID")

#given the apogee_id, get a dict of teff, logg, met
def get_star(star_id, cluster_name):
    star = {}
    cluster = clusters_stars.loc["CLUSTER", cluster_name]
    s = cluster.loc['APOGEE_ID', star_id]
    star['teff'] = s["TEFF"]
    star['logg'] = s["LOGG"]
    star['met'] = s["M_H"]
    return star

#given a cluster and model, returns a table of all the stars with lum, age, teff calculated with that model
def get_cluster_table(cluster_name, writefile=False, verbose=False):
    if os.path.isfile(f'{cluster_name}.fits'):
        if verbose:
            print(f'{cluster_name} has a pre-made table, nice!')
        stars_table = Table.read(f'{cluster_name}.fits', format='fits')
        stars_table.meta['cluster'] = cluster_name
        stars_table.add_index('model')
        return stars_table
    if (cluster_name not in clusters_stars['CLUSTER']):
        print(f'{cluster_name} is not in the OCCAM database')
        if verbose:
            print(f'Please choose from the following list: ')
            for i in clusters_stars['CLUSTER']:
                print(i, end=", ")
        return
    print(f'{cluster_name} does not have a pre-made table. This will take a few minutes.')
    gridsearch_fit_setup()
        
    cluster = clusters_stars.loc["CLUSTER", cluster_name] #table of all the stars in the cluster
    cluster_ids = cluster["APOGEE_ID"] 
    stars_dict = {'err':[], 'eep':[], 'age':[], 'lum':[], 'model':[], 'mass':[], 'teff':[]}
    # add the columns that you saved in clusters_stars
    for i in cluster.columns:
        stars_dict[i] = []
    
    idx = 0
    for ID in cluster_ids:
        #apogee_star contains LOGG, TEFF, M/H, APOGEE_ID, and any other data you saved into occam_dr17_stars
        apogee_star = cluster.loc['APOGEE_ID', ID]
        if verbose: 
            print(idx, ID, end=' ')
        idx+=1
        for name, model in models_dict.items():
            if (ID in skip[name]):
                if verbose:
                    print(f'{name}: Skipping star', end=' ')
                continue
            if verbose:
                print(f'{name}', end='')
            star, fit = models_dict[name].gridsearch_fit(get_star(ID, cluster_name), scale=(1000, 1, 0.1), tol=1e-6, verbose=False)
            stars_dict['teff'].append(star['teff'])
            stars_dict['lum'].append(np.log10(star['lum']))
            stars_dict['age'].append(star['age'])
            stars_dict['model'].append(name)
            stars_dict['eep'].append(star['eep'])
            stars_dict['mass'].append(star['mass'])
            stars_dict['ID'].append(ID)
            stars_dict['err'].append(int(fit.fun>1e-6))
            
            for i in cluster_columns:
                stars_dict[i].append(apogee_star[i])
                
            if verbose: 
                if (fit.fun > 1e-6):
                        print(f' not fit within tolerance')
                else:
                        print("")
    stars_table = Table(stars_dict, meta={'cluster':cluster})
    if writefile:
        stars_table.write(f'{cluster_name}.fits', format='fits', overwrite=True)
    return stars_table


#splits cluster_table into 4 tables, each with only the data for the associated model
def get_model_tables(cluster_table, models=['yrec','dart','mist','gars']):
    tables = []
    cluster_table.add_index('model')
    for model_name in models:
        model_table = cluster_table.loc['model', model_name]
        tables.append(model_table)
    return tables


'''Setup for gridsearch_fit'''
def gridsearch_fit_setup():
    # use grid points between ZAMS (201) and RGBump (605)
    qstring = '0.6 <= initial_mass <= 2 and -1.0 <= initial_met <= 0.5 and 201 <= eep <= 605'

    # Whether to fit evolved metallicity (True) or use the initial metallicity.
    # False is probably fine if you're not on the giant branch.
    evolve_met = False
    
    global yrec
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

    global mist
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

    global dart
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

    global gars
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
    
    global models_dict
    models_dict = {'yrec':yrec, 'mist':mist, 'dart':dart, 'gars':gars}