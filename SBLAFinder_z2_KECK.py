#%% Packages, functions and data

import matplotlib.pyplot as plt
import h5py
import numpy as np
from tqdm import trange, tqdm
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import z_at_value
from collections import Counter
import spectres
from joblib import Parallel, delayed
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)

plt.rcParams["font.family"] = "serif"
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17

def restframification(wave,z):
    wave_rest = wave/(1+z)
    return wave_rest

def rebin(a, *args):
    # a=array(a)
#http://scipy-cookbook.readthedocs.io/items/Rebinning.html
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = a.shape
    lenShape = len(shape)
    factor =  np.asarray(shape)/np.asarray(args)
    if int(factor[-1])!=factor[-1]:  ###### This if statment is to deal with length issues and is not orginal
        for i,row in enumerate(a):
            a[i]=a[i][:int(args[-1])*int(factor)]
        shape=a.shape
    evList = ['a.reshape('] + \
             ['int(args[%d]),int(factor[%d]),'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
             ['/factor[%d]'%i for i in range(lenShape)]

    return (eval(''.join(evList)))

def sub_redshift(posz, z_front, z_max):
    front_of_box_z = z_front # Redshift at the front of box
    h = 0.6774 # h parameter
    kpc_to_mpc = 1e-3 # Difference between ckpc to cMpc
    ls_distance = (posz/h) * kpc_to_mpc # Physical cMpc
    z = np.zeros(len(ls_distance))

    init_distance = cosmo.comoving_distance(front_of_box_z).to_value()

    for i in range(len(ls_distance)):
        dist = (init_distance + ls_distance[i]) * u.Mpc
        z[i] = dist.to(cu.redshift, cu.redshift_distance(cosmo, kind="comoving", zmax=z_max)).to_value()
    return z

def peculiar_redshift(vel,z):
    #Peculiar velocity comes in km/s
    c = 2.99792e5 #km/s
    dz = vel/c * (1+z)
    return dz

def v_dijkstra04(mass,z,omega_m,h):
    v = 18.4 * np.power(1/1e8 * mass,1/3) * np.power((1+z)/12,1/2) * np.power((omega_m*h**2)/0.135,1/6) #km/s
    return v

def deltav(z_abs,z_em):
    c = 2.99792e5 #km/s
    v = c * (abs(z_abs - z_em))/(1 + z_em)
    return v

def makeMatrix(xVec,yVec,cellBin): #CONTOUR MAKER
   
    #Contours
    xx  = np.linspace(np.min(xVec),np.max(xVec),cellBin)
    yy  = np.linspace(np.min(yVec),np.max(yVec),cellBin)
    X, Y = np.meshgrid(xx,yy)
    Z = np.zeros((len(yy),len(xx)))
    for i in range(len(xx)-1):
        for j in range(len(yy)-1):
            nObj = (xVec>xx[i]) & (xVec<xx[i+1]) & (yVec>yy[j]) & (yVec<yy[j+1])
            Z[j][i] = np.log10(len(xVec[nObj]))
    return X, Y, Z
    
def wave_rebin(bins, wavelength):
    wavelength_rebin = np.zeros(int(len(wavelength)/bins))

    i = 0
    wi = 0
    while wi < len(wavelength_rebin):
        wavelength_rebin[wi] = np.sum(wavelength[i:i+bins])/bins
        wi += 1
        i += bins

    lya = np.where((wavelength_rebin > 3640) & (wavelength_rebin < 3648)) # Lyman alpha at z=2. is 3647.01 Angstroms
    
    wave1 = wavelength_rebin[lya][-1]
    wave2 = wavelength_rebin[lya][-2]
    
    c = 2.99792e5 #km/s
    wave_lya = 1215.670 #Angstrom
    z1 = wave1/wave_lya - 1
    #z2 = wave2/wave_lya - 1
    dz = (wave1 - wave2)/wave_lya
    deltav = c * dz/(1+z1)
    
    return wavelength_rebin, deltav

    
f = ascii.read('SubhaloInfo.csv',format='csv')

subhalo_posx = np.array(f['Subhalo_PosX']) # Subhalo positions in x
subhalo_posy = np.array(f['Subhalo_PosY']) # Subhalo positions in y
subhalo_posz = np.array(f['Subhalo_PosZ']) # Subhalo positions in z
subhalo_mass = np.array(f['Subhalo_Mass']) # Subhalo mass. To convert to physical mass you have to multiply by 1e10/H0
subhalo_radhm = np.array(f['Subhalo_HalfMassRadius']) # Subhalo half mass radius. Twice this is the virial radius.
subhalo_z = np.array(f['Subhalo_GasMetal']) # Subhalo gas metallicity
subhalo_vz = np.array(f['Subhalo_PVZ']) # Subhalo peculiar velocity in z axis (km/s)
subhalo_vdisp = np.array(f['Subhalo_VDispersion']) # Subhalo velocity dispersion (km/s)
subhalo_vmax = np.array(f['Subhalo_VMax']) # Subhalo maximum velocity of the rotation curve (km/s)

f = ascii.read('GroupInfo.csv',format='csv')

group_posx = np.array(f['Group_CMX']) # Group positions in x
group_posy = np.array(f['Group_CMY']) # Group positions in y
group_posz = np.array(f['Group_CMZ']) # Group positions in z
group_mass = np.array(f['Group_Mass']) # Group Mass
group_z = np.array(f['Group_Metal']) # Group Metallicity
group_vrad = np.array(f['Group_RCrit200']) # Group virial radius
group_subhaloid = np.array(f['Subhalo_ID']) # Central Subhalo ID

mass_filter = (np.log10(group_mass*1e10/0.6774) > 9.05)

#f = h5py.File('spectra_TNG50-1_z3.0_n1000d2-fullbox_SDSS-BOSS_HI_combined.hdf5', 'r')
wavelength = np.array(f['wave']) 

#%% Wavelength rebins

wave_rebins = [wave_rebin(11, wavelength)[0], wave_rebin(12, wavelength)[0], wave_rebin(18, wavelength)[0], wave_rebin(25, wavelength)[0], wave_rebin(31, wavelength)[0],\
              wave_rebin(40, wavelength)[0], wave_rebin(47, wavelength)[0], wave_rebin(55, wavelength)[0], wave_rebin(61, wavelength)[0], wave_rebin(67, wavelength)[0],\
              wave_rebin(74, wavelength)[0],wave_rebin(82, wavelength)[0],wave_rebin(88, wavelength)[0]]

vel = [wave_rebin(11, wavelength)[1], wave_rebin(12, wavelength)[1], wave_rebin(18, wavelength)[1], wave_rebin(25, wavelength)[1], wave_rebin(31, wavelength)[1],\
              wave_rebin(40, wavelength)[1], wave_rebin(47, wavelength)[1], wave_rebin(55, wavelength)[1], wave_rebin(61, wavelength)[1], wave_rebin(67, wavelength)[1],\
          wave_rebin(74, wavelength)[1],wave_rebin(82, wavelength)[1],wave_rebin(88, wavelength)[1]]
    
#%%

def dla_finder(wave, flux):
	print('Multiple wavelength solutions. There are:', len(wave), 'wavelength solutions')
	alldla_index = []
	alldlawave_index = []
	
	for irebin in tqdm(range(len(wave))):
		dlaindex = []
		dlawave_index = []
		
		for k in range(50):
			start = int(len(flux)/50*k)
			end = int(len(flux)/50*(k+1))
			
			flux_rebin = spectres.spectres(wave[irebin],wavelength,flux[start:end],verbose=False)
			
			for i in range(len(flux_rebin)):
				for j in range(len(flux_rebin[i])):
					if flux_rebin[i][j] <= 0.001 and np.mean(flux_rebin[i][j-2:j+2]) <= 0.01:
						for f in range(5):
							dlaindex.append(i + start)
						dlawave_index.append(j-2)
						dlawave_index.append(j-1)
						dlawave_index.append(j)
						dlawave_index.append(j+1)
						dlawave_index.append(j+2)
			
		dlaindex = np.array(dlaindex)
		dlawave_index = np.array(dlawave_index)
			
		alldla_index.append(dlaindex)	
		alldlawave_index.append(dlawave_index)
	
	return alldla_index, alldlawave_index	

dlas = dla_finder(wave_rebins, f['flux'])

file = Table([dlas[0],dlas[1]],names=('DLA Index', 'DLA Wavelength Index'))
file.write('DLA_SBLA_HighRes/DLA_AllBins_Index.fits',overwrite=True)

with fits.open('DLA_SBLA_HighRes/DLA_AllBins_Index.fits') as hdul:
    dla_info = hdul[1].data

#%%

def sbla_finder(flux, wave, dla, f_value):
    print('Multiple wavelength solutions. There are:', len(wave), 'wavelength solutions')

    allsbla_index = []
    allsblawave_index = []

    # Loop over `f` in parallel using threads instead of processes to avoid pickling errors
    def process_wave_solution(f):
        sbla_index = []
        sblawave_index = []
        
        # Cache dla indices for the current `f`
        dla_f = np.array(dla[f][0])
        dla_wave_f = np.array(dla[f][1])

        for k in trange(50):
            # Calculate the rebinning range
            start = int(len(flux)/50*k)
            end = int(len(flux)/50*(k+1))

            # Rebin flux
            flux_rebin = spectres.spectres(wave[f], wavelength, flux[start:end], verbose=False)

            # Apply conditions to find `find_sbla`
            condition = (flux_rebin < f_value) & (wave[f] > 1215.670 * (1 + 2))
            find_sbla = np.where(condition)

            # Remove `dla` indices
            find_sbla_indices = find_sbla[0] + start
            to_remove = np.in1d(find_sbla_indices, dla_f)
            
            sbla_index_k = np.delete(find_sbla_indices, to_remove)
            sblawave_index_k = np.delete(find_sbla[1], to_remove)
            
            sbla_index.extend(sbla_index_k)
            sblawave_index.extend(sblawave_index_k)

        # Remove any indices that overlap with `dla`
        sbla_index = np.array(sbla_index)
        sblawave_index = np.array(sblawave_index)

        valid_indices = np.isin(sbla_index, dla_f, invert=True)
        sbla_nodla = sbla_index[valid_indices]
        sbla_waveindex_nodla = sblawave_index[valid_indices]

        return sbla_nodla, sbla_waveindex_nodla

    # Process all wavelength solutions in parallel using threads
    results = Parallel(n_jobs=-1, prefer="threads")(delayed(process_wave_solution)(f) for f in tqdm(range(len(wave))))

    # Combine results
    for sbla_nodla, sbla_waveindex_nodla in results:
        allsbla_index.append(sbla_nodla)
        allsblawave_index.append(sbla_waveindex_nodla)
    
    return allsbla_index, allsblawave_index

sbla025 = sbla_finder(f['flux'], wave_rebins, dla_info, 0.25)

file = Table([sbla025[0],sbla025[1]],names=('SBLA Index', 'SBLA Wavelength Index'))
file.write('DLA_SBLA_HighRes/SBLA_025_AllBins_Index.fits',overwrite=True)

sbla015 = sbla_finder(f['flux'], wave_rebins, dla_info, 0.15)

file = Table([sbla015[0],sbla015[1]],names=('SBLA Index', 'SBLA Wavelength Index'))
file.write('DLA_SBLA_HighRes/SBLA_015_AllBins_Index.fits',overwrite=True)

sbla005 = sbla_finder(f['flux'], wave_rebins, dla_info, 0.05)

file = Table([sbla005[0],sbla005[1]],names=('SBLA Index', 'SBLA Wavelength Index'))
file.write('DLA_SBLA_HighRes/SBLA_005_AllBins_Index.fits',overwrite=True)

#%%

with fits.open('DLA_SBLA_HighRes/SBLA_025_AllBins_Index.fits') as hdul:
    sbla_025 = hdul[1].data

with fits.open('DLA_SBLA_HighRes/SBLA_015_AllBins_Index.fits') as hdul:
    sbla_015 = hdul[1].data

with fits.open('DLA_SBLA_HighRes/SBLA_005_AllBins_Index.fits') as hdul:
    sbla_005 = hdul[1].data

#%% Estimation of SBLAs in Halos

# DLAs and SBLAs were already detected. Let's now find how many are in halos!

def sbla_halo_finder(file, posx, posy, posz, vrad, pecz, mass, vdisp, vmax, wavelength,\
                     spec_file, zfront, zmax, flim, pxl_list):
    # Initialising variables with all the info
    sblas_in_cgm_vrad_vdisp_all = []
    rad_idx_vrad_vdisp_all = []
    sblas_in_cgm_vrad_vmax_all = []
    rad_idx_vrad_vmax_all = []
    
    with fits.open(file) as hdul:
        sbla = hdul[1].data
        
    # Iterating over the number of rebins in the wavelength
    for iwave in trange(len(sbla)):
        # Reading the file with all detected SBLAs (no DLAs)
        # with fits.open(file[iwave]) as hdul:
        #     data = hdul[1].data
        #     sbla_nodla = data['SBLA Index']
        #     sbla_waveidx_nodla = data['SBLA Wavelength Index']
        
        # Mass filter for this sample
        mass_filter = (np.log10(mass*1e10/0.6774) > 9.05)
        
        # Calculating redshift and peculiar redshift of each halo
        group_redshift = sub_redshift(posz[mass_filter], zfront, zmax)
        group_pecz = np.array(peculiar_redshift(pecz[group_subhaloid][mass_filter], group_redshift))
        
        # Calculating redshift of each SBLA
        wave_lyalpha = 1215.670
        sbla_redshift = []
        # for k in tqdm(range(len(sbla))):
        x = (wavelength[iwave][sbla[iwave][1]] - wave_lyalpha)/wave_lyalpha
        sbla_redshift.append(x)
    # sbla_redshift = (wavelength[iwave][sbla_waveidx_nodla]-wave_lyalpha)/wave_lyalpha
    
        # Variables that contain the position of the spectra in the TNG50 box
        ray_pos_x = spec_file['ray_pos'][:, 0][sbla[iwave][0]]
        ray_pos_y = spec_file['ray_pos'][:, 1][sbla[iwave][0]]
        
        # Filtered variables according to the mass filter applied
        group_posx_filtered = posx[mass_filter]
        group_posy_filtered = posy[mass_filter]
        group_vrad_filtered = vrad[mass_filter]
        group_pecz_filtered = group_pecz
        group_redshift_filtered = group_redshift
        subhalo_vdisp_filtered = vdisp[group_subhaloid][mass_filter]
        subhalo_vmax_filtered = vmax[group_subhaloid][mass_filter]
        
        # "Local" variables that store SBLAs that are inside halos, both for
        # velocity dispersion
        sblas_in_cgm_vrad_vdisp = []
        rad_idx_vrad_vdisp = []
        # maximum velocity
        sblas_in_cgm_vrad_vmax = []
        rad_idx_vrad_vmax = []
        
        # Loop over the filtered positions
        for i in range(len(group_posx_filtered)):
            
            # Compute the squared distance only once
            delta_x = ray_pos_x - group_posx_filtered[i]
            delta_y = ray_pos_y - group_posy_filtered[i]
            distance_squared = delta_x**2 + delta_y**2
            
            # Apply the distance mask
            distance_mask = distance_squared <= group_vrad_filtered[i]**2

            # Compute the velocity difference only once
            velocity_diff = deltav(sbla_redshift, group_pecz_filtered[i] + group_redshift_filtered[i])
            
            # Apply the velocity mask
            velocity_mask = velocity_diff <= subhalo_vdisp_filtered[i]
            velocity_mask_max = velocity_diff <= subhalo_vmax_filtered[i]
            
            # Combine the two masks using logical AND
            final_mask = distance_mask & velocity_mask
            final_mask_vmax = distance_mask & velocity_mask_max
            
            # Use np.nonzero instead of np.where to get indices directly
            finder_vrad = np.nonzero(final_mask[0])[0]
            finder_vrad_vmax = np.nonzero(final_mask_vmax[0])[0]
        
            # If any indices found, append the results
            if finder_vrad.size > 0:
                rad_idx_vrad_vdisp.append(i)
                sblas_in_cgm_vrad_vdisp.append(finder_vrad)
                
            # If any indices found, append the results
            if finder_vrad_vmax.size > 0:
                rad_idx_vrad_vmax.append(i)
                sblas_in_cgm_vrad_vmax.append(finder_vrad_vmax)
                
        sblas_in_cgm_vrad_vdisp_all.append(sblas_in_cgm_vrad_vdisp)
        rad_idx_vrad_vdisp_all.append(rad_idx_vrad_vdisp)
        sblas_in_cgm_vrad_vmax_all.append(sblas_in_cgm_vrad_vmax)
        rad_idx_vrad_vmax_all.append(rad_idx_vrad_vmax)
    
    for iwave in range(len(sbla)):
        file = Table([rad_idx_vrad_vmax_all[iwave],sblas_in_cgm_vrad_vmax_all[iwave]],names=('Subhalo Index', 'SBLA Indices'))
        file.write('Index_HighRes/SBLA'+flim+'_'+pxl_list[iwave]+'_vmax_vrad.fits',overwrite=True)
        
        file = Table([rad_idx_vrad_vdisp_all[iwave],sblas_in_cgm_vrad_vdisp_all[iwave]],names=('Subhalo Index', 'SBLA Indices'))
        file.write('Index_HighRes/SBLA'+flim+'_'+pxl_list[iwave]+'_vdisp_vrad.fits',overwrite=True)
        
    # return rad_idx_vrad_vmax_all, sblas_in_cgm_vrad_vmax_all, rad_idx_vrad_vdisp_all, sblas_in_cgm_vrad_vdisp_all

vel_str = [str(round(elements,0)) for elements in vel]
t = sbla_halo_finder('DLA_SBLA_HighRes/SBLA_025_AllBins_Index.fits', group_posx, group_posy, group_posz, group_vrad, subhalo_vz, group_mass,\
                      subhalo_vdisp, subhalo_vmax, wave_rebins, f,  2.0020281392528516, 2.1, '025',\
                        vel_str)

t = sbla_halo_finder('DLA_SBLA_HighRes/SBLA_015_AllBins_Index.fits', group_posx, group_posy, group_posz, group_vrad, subhalo_vz, group_mass,\
                      subhalo_vdisp, subhalo_vmax, wave_rebins, f,  2.0020281392528516, 2.1, '015',\
                        vel_str)
    
t = sbla_halo_finder('DLA_SBLA_HighRes/SBLA_005_AllBins_Index.fits', group_posx, group_posy, group_posz, group_vrad, subhalo_vz, group_mass,\
                      subhalo_vdisp, subhalo_vmax, wave_rebins, f,  2.0020281392528516, 2.1, '005',\
                        vel_str)

#%% SBLAs in CGM

import pandas as pd

sbla025_vmax_vrad = {}
sbla015_vmax_vrad = {}
sbla005_vmax_vrad = {}

for i in trange(len(vel_str)):
    with fits.open('Index_HighRes/SBLA025_'+vel_str[i]+'_vmax_vrad.fits') as hdul:
        data = hdul[1].data
        t1 = data['Subhalo Index']
        t2 = data['SBLA Indices']

    if i == 0:
        sbla025_vmax_vrad['Halo Index'] = [t1]
        sbla025_vmax_vrad['SBLAs Indices'] = [t2]
    else:
        sbla025_vmax_vrad['Halo Index'].append(t1)
        sbla025_vmax_vrad['SBLAs Indices'].append(t2)

    with fits.open('Index_HighRes/SBLA015_'+vel_str[i]+'_vmax_vrad.fits') as hdul:
            data = hdul[1].data
            t1 = data['Subhalo Index']
            t2 = data['SBLA Indices']
    
    if i == 0:
        sbla015_vmax_vrad['Halo Index'] = [t1]
        sbla015_vmax_vrad['SBLAs Indices'] = [t2]
    else:
        sbla015_vmax_vrad['Halo Index'].append(t1)
        sbla015_vmax_vrad['SBLAs Indices'].append(t2)

    with fits.open('Index_HighRes/SBLA005_'+vel_str[i]+'_vmax_vrad.fits') as hdul:
            data = hdul[1].data
            t1 = data['Subhalo Index']
            t2 = data['SBLA Indices']
    
    if i == 0:
        sbla005_vmax_vrad['Halo Index'] = [t1]
        sbla005_vmax_vrad['SBLAs Indices'] = [t2]
    else:
        sbla005_vmax_vrad['Halo Index'].append(t1)
        sbla005_vmax_vrad['SBLAs Indices'].append(t2)

sbla025_vdisp_vrad = {}
sbla015_vdisp_vrad = {}
sbla005_vdisp_vrad = {}

for i in trange(len(vel_str)):
    with fits.open('Index_HighRes/SBLA025_'+vel_str[i]+'_vdisp_vrad.fits') as hdul:
        data = hdul[1].data
        t1 = data['Subhalo Index']
        t2 = data['SBLA Indices']

    if i == 0:
        sbla025_vdisp_vrad['Halo Index'] = [t1]
        sbla025_vdisp_vrad['SBLAs Indices'] = [t2]
    else:
        sbla025_vdisp_vrad['Halo Index'].append(t1)
        sbla025_vdisp_vrad['SBLAs Indices'].append(t2)

    with fits.open('Index_HighRes/SBLA015_'+vel_str[i]+'_vdisp_vrad.fits') as hdul:
            data = hdul[1].data
            t1 = data['Subhalo Index']
            t2 = data['SBLA Indices']
    
    if i == 0:
        sbla015_vdisp_vrad['Halo Index'] = [t1]
        sbla015_vdisp_vrad['SBLAs Indices'] = [t2]
    else:
        sbla015_vdisp_vrad['Halo Index'].append(t1)
        sbla015_vdisp_vrad['SBLAs Indices'].append(t2)

    with fits.open('Index_HighRes/SBLA005_'+vel_str[i]+'_vdisp_vrad.fits') as hdul:
            data = hdul[1].data
            t1 = data['Subhalo Index']
            t2 = data['SBLA Indices']
    
    if i == 0:
        sbla005_vdisp_vrad['Halo Index'] = [t1]
        sbla005_vdisp_vrad['SBLAs Indices'] = [t2]
    else:
        sbla005_vdisp_vrad['Halo Index'].append(t1)
        sbla005_vdisp_vrad['SBLAs Indices'].append(t2)
            
sbla025_vmax_vrad = pd.DataFrame.from_dict(sbla025_vmax_vrad)
sbla015_vmax_vrad = pd.DataFrame.from_dict(sbla015_vmax_vrad)
sbla005_vmax_vrad = pd.DataFrame.from_dict(sbla005_vmax_vrad)

sbla025_vdisp_vrad = pd.DataFrame.from_dict(sbla025_vdisp_vrad)
sbla015_vdisp_vrad = pd.DataFrame.from_dict(sbla015_vdisp_vrad)
sbla005_vdisp_vrad = pd.DataFrame.from_dict(sbla005_vdisp_vrad)

#%% SBLA Properties estimation

# SBLA PROPERTIES
def sbla_prop(file, sblas_files, halo_files, wavelengths, spec_binning, mass, prob,directory):
    with fits.open(file) as hdul:
        data = hdul[1].data
    
    for i in trange(len(wavelengths)):
                    
        lya = np.where((wavelengths[i] > 1213*(1+2)) & (wavelengths[i] <= 1218*(1+2)))  
        wave1 = wavelengths[i][lya][-1]
        wave2 = wavelengths[i][lya][-2]
        wave1 - wave2

        bin_list = []
        for k in range(len(data[i]['SBLA Index'])):
            bin_list.append(spec_binning[i])
        
        spec_num = data[i]['SBLA Index']
        wave_idx = data[i]['SBLA Wavelength Index']
        posx = f['ray_pos'][:,0][data[i]['SBLA Index']]
        posy = f['ray_pos'][:,1][data[i]['SBLA Index']]
        wave_c = wavelengths[i][data[i]['SBLA Wavelength Index']]
        wave_min = wavelengths[i][data[i]['SBLA Wavelength Index']] - (wave1 - wave2)
        wave_max = wavelengths[i][data[i]['SBLA Wavelength Index']] + (wave1 - wave2)

        
        sblas_array = np.concatenate(sblas_files[i])
        unique_sblas_indices = np.unique(sblas_array)
        
        
        sblas_index_dict = {index: True for index in unique_sblas_indices}
        
        hidx = []
        for k in range(len(sblas_files[i])):
            x = len(sblas_files[i][k])
            for q in range(x):
                hidx.append(halo_files[i][k])
        
        
        conversion_factor = 1e10 / 0.6774
        mass_values = np.log10(mass[mass_filter][hidx] * conversion_factor)
        
        log_mass_boundaries_next = [9.0, 9.4, 9.8, 10.2, 10.6, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0]
        bin_indices = np.searchsorted(log_mass_boundaries_next, mass_values, side='left')
        
        num_bins = 11
        bin_array = np.zeros((len(data[i]['SBLA Index']), num_bins), dtype=int)
        
        for idx in range(len(unique_sblas_indices)):
            sbla_index = unique_sblas_indices[idx]
            
            if sbla_index in sblas_index_dict:
                positions = np.where(sblas_array == sbla_index)[0]
        
                for pos in positions:
                    mass_value = mass_values[pos]
                    bin_index = np.searchsorted(log_mass_boundaries_next, mass_value, side='left')
                    if bin_index < num_bins:
                        bin_array[sbla_index, bin_index] = 1
        
        bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10, bin11 = bin_array.T

        if i < 1:
            file_fits = Table([bin_list, spec_num, wave_idx, posx, posy, wave_c, wave_min, wave_max, bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10, bin11],\
                         names=('Spectral Binning', 'Index of Spectrum', 'Index of Wavelength', 'X Position [ckpc/h]', 'Y Position [ckpc/h]', 'SBLA Central Wavelength [A]',\
                                'SBLA Minimum Wavelength [A]', 'SBLA Maximum Wavelength [A]', 'SBLA in Mass Bin [9,9.4]', 'SBLA in Mass Bin [9.4,9.8]','SBLA in Mass Bin [9.8,10.2]',\
                               'SBLA in Mass Bin [10.2,10.6]', 'SBLA in Mass Bin [10.6,11.0]', 'SBLA in Mass Bin [11,11.4]','SBLA in Mass Bin [11.4,11.8]','SBLA in Mass Bin [11.8,12.2]',\
                               'SBLA in Mass Bin [12.2,12.6]','SBLA in Mass Bin [12.6,13]','SBLA in Mass Bin [>13]'))
            file_fits.write(directory+'SBLAProps_'+prob+'.fits',overwrite=True)
        else:
            with fits.open(directory+'/SBLAProps_'+prob+'.fits') as hdul:
                data_fits = hdul[1].data
            
            bin_list = np.concatenate((data_fits['Spectral Binning'], bin_list))
            spec_num = np.concatenate((data_fits['Index of Spectrum'], spec_num))
            wave_idx = np.concatenate((data_fits['Index of Wavelength'], wave_idx))
            posx = np.concatenate((data_fits['X Position [ckpc/h]'], posx))
            posy = np.concatenate((data_fits['Y Position [ckpc/h]'], posy))
            wave_c = np.concatenate((data_fits['SBLA Central Wavelength [A]'], wave_c))
            wave_min = np.concatenate((data_fits['SBLA Minimum Wavelength [A]'], wave_min))
            wave_max = np.concatenate((data_fits['SBLA Maximum Wavelength [A]'], wave_max))
            bin1 = np.concatenate((data_fits['SBLA in Mass Bin [9,9.4]'], bin1))
            bin2 = np.concatenate((data_fits['SBLA in Mass Bin [9.4,9.8]'], bin2))
            bin3 = np.concatenate((data_fits['SBLA in Mass Bin [9.8,10.2]'], bin3))
            bin4 = np.concatenate((data_fits['SBLA in Mass Bin [10.2,10.6]'], bin4))
            bin5 = np.concatenate((data_fits['SBLA in Mass Bin [10.6,11.0]'], bin5))
            bin6 = np.concatenate((data_fits['SBLA in Mass Bin [11,11.4]'], bin6))
            bin7 = np.concatenate((data_fits['SBLA in Mass Bin [11.4,11.8]'], bin7))
            bin8 = np.concatenate((data_fits['SBLA in Mass Bin [11.8,12.2]'], bin8))
            bin9 = np.concatenate((data_fits['SBLA in Mass Bin [12.2,12.6]'], bin9))
            bin10 = np.concatenate((data_fits['SBLA in Mass Bin [12.6,13]'], bin10))
            bin11 = np.concatenate((data_fits['SBLA in Mass Bin [>13]'], bin11))

            file_fits = Table([bin_list, spec_num, wave_idx, posx, posy, wave_c, wave_min, wave_max, bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10, bin11],\
                         names=('Spectral Binning', 'Index of Spectrum', 'Index of Wavelength', 'X Position [ckpc/h]', 'Y Position [ckpc/h]', 'SBLA Central Wavelength [A]',\
                                'SBLA Minimum Wavelength [A]', 'SBLA Maximum Wavelength [A]', 'SBLA in Mass Bin [9,9.4]', 'SBLA in Mass Bin [9.4,9.8]','SBLA in Mass Bin [9.8,10.2]',\
                               'SBLA in Mass Bin [10.2,10.6]', 'SBLA in Mass Bin [10.6,11.0]', 'SBLA in Mass Bin [11,11.4]','SBLA in Mass Bin [11.4,11.8]','SBLA in Mass Bin [11.8,12.2]',\
                               'SBLA in Mass Bin [12.2,12.6]','SBLA in Mass Bin [12.6,13]','SBLA in Mass Bin [>13]'))
            file_fits.write(directory +'SBLAProps_'+prob+'.fits',overwrite=True)

fnames = ['025_vmax', '025_vdisp']

t = sbla_prop('DLA_SBLA_HighRes/SBLA_025_AllBins_Index.fits', sbla025_vmax_vrad['SBLAs Indices'], sbla025_vmax_vrad['Halo Index'],\
              wave_rebins, vel_str, group_mass, fnames[0],'DLA_SBLA_HighRes/')
t = sbla_prop('DLA_SBLA_HighRes/SBLA_025_AllBins_Index.fits', sbla025_vdisp_vrad['SBLAs Indices'], sbla025_vdisp_vrad['Halo Index'],\
              wave_rebins, vel_str, group_mass, fnames[1],'DLA_SBLA_HighRes/')

fnames = ['015_vmax', '015_vdisp']
    
t = sbla_prop('DLA_SBLA_HighRes/SBLA_015_AllBins_Index.fits', sbla015_vmax_vrad['SBLAs Indices'], sbla015_vmax_vrad['Halo Index'],\
              wave_rebins, vel_str, group_mass, fnames[0],'DLA_SBLA_HighRes/')
t = sbla_prop('DLA_SBLA_HighRes/SBLA_015_AllBins_Index.fits', sbla015_vdisp_vrad['SBLAs Indices'], sbla015_vdisp_vrad['Halo Index'],\
              wave_rebins, vel_str, group_mass, fnames[1],'DLA_SBLA_HighRes/')
    
fnames = ['005_vmax', '005_vdisp']
    
t = sbla_prop('DLA_SBLA_HighRes/SBLA_005_AllBins_Index.fits', sbla005_vmax_vrad['SBLAs Indices'], sbla005_vmax_vrad['Halo Index'],\
              wave_rebins, vel_str, group_mass, fnames[0],'DLA_SBLA_HighRes/')
t = sbla_prop('DLA_SBLA_HighRes/SBLA_005_AllBins_Index.fits', sbla005_vdisp_vrad['SBLAs Indices'], sbla005_vdisp_vrad['Halo Index'],\
              wave_rebins, vel_str, group_mass, fnames[1],'DLA_SBLA_HighRes/')
    
    
#%% Hierarchical SBLA

# with fits.open('DLA_SBLA_z3/SBLAProps_025_vmax.fits') as hdul:
#     sblaprops_vmax_025 = hdul[1].data
# with fits.open('DLA_SBLA_z3/SBLAProps_015_vmax.fits') as hdul:
#     sblaprops_vmax_015 = hdul[1].data
# with fits.open('DLA_SBLA_z3/SBLAProps_005_vmax.fits') as hdul:
#     sblaprops_vmax_005 = hdul[1].data

# with fits.open('DLA_SBLA_z3/SBLAProps_025_vdisp.fits') as hdul:
#     sblaprops_vdisp_025 = hdul[1].data
# with fits.open('DLA_SBLA_z3/SBLAProps_015_vdisp.fits') as hdul:
#     sblaprops_vdisp_015 = hdul[1].data
# with fits.open('DLA_SBLA_z3/SBLAProps_005_vdisp.fits') as hdul:
#     sblaprops_vdisp_005 = hdul[1].data

# def hierarchical_sbla_maker(sblaprop_file,directory,flim):
#     """
#     Generates a hierarchical SBLA table by identifying non-eaten SBLAs based on their spectral binning.

#     Parameters:
#     -----------
#     sblaprop_file : FITS_rec
#         Input FITS file containing SBLA properties.
#     directory: str
#         String that incorporates directory to save file in
#     flim : str
#         String suffix that dictates flux limit

#     Returns:
#     --------
#     None. Writes the hierarchical SBLA table to a .fits file in the wanted directory
#     """
#     checked = set()  # Using a set for faster lookups
#     idx_saver = []
    
#     # Pre-extract other columns
#     index_of_spectrum = sblaprop_file['Index of Spectrum']
#     central_wavelength = sblaprop_file['SBLA Central Wavelength [A]']
#     min_wavelength = sblaprop_file['SBLA Minimum Wavelength [A]']
#     max_wavelength = sblaprop_file['SBLA Maximum Wavelength [A]']
    
#     for i in trange(len(index_of_spectrum)):
#         current_spectrum = index_of_spectrum[i]
    
#         if current_spectrum not in checked:
#             checked.add(current_spectrum)
    
#             # Find indices for the current spectrum
#             finder = np.where(index_of_spectrum == current_spectrum)[0]
    
#             # Conversion of spectral binning to floats
#             specbinnings = np.zeros(len(finder))
#             for k in range(len(finder)):
#                 specbinnings[k] = float(sblaprop_file['Spectral Binning'][finder][k])
    
#             # Highest binning velocity
#             max_specbin = np.max(specbinnings)
#             # SBLAs with the highest binning velocity
#             max_bins = np.where(specbinnings == max_specbin)[0]
    
#             eaten = set()
#             for k in range(len(finder) - 1, 0, -1):  # Counting the list backwards in order of going from highest binning to lowest
#                 bin_k = specbinnings[k]
    
#                 # Determining condition that finds SBLAs that "eat" other SBLAS, where we have:
#                 # 1) SBLAs have to be inside of the wavelength range of a higher order SBLA
#                 # 2) SBLAs cannot "eat" themselves
#                 # 3) Only higher binning SBLAs can eat lower order SBLAs
#                 condition = (central_wavelength[finder] > min_wavelength[finder[k]]) & \
#                             (central_wavelength[finder] < max_wavelength[finder[k]]) & \
#                             (specbinnings < bin_k) & \
#                             (np.arange(len(finder)) != k)
    
#                 eat_condition = np.where(condition)[0]
    
#                 # Ensuring that we are not double counting SBLAs nor counting the higher binning SBLAs
#                 for j in eat_condition:
#                     if finder[j] not in eaten and finder[j] not in max_bins:
#                         eaten.add(finder[j])
    
#             # Saving all indices with non-eaten SBLAs
#             idx_saver.extend([j for j in finder if j not in eaten])
    
#     # Creating hierarchical SBLA Table
#     htable = sblaprop_file[np.sort(idx_saver)]
#     htable.write(directory+'SBLAProps_Hierarchical'+flim+'.fits',overwrite=True)
    
# t = hierarchical_sbla_maker(sblaprops_vmax_025,\
#                             '/home/dsantos/Desktop/SBLA/DylanSimulations/DLA_SBLA_z3','_025_vmax')
# t = hierarchical_sbla_maker(sblaprops_vmax_015,\
#                             '/home/dsantos/Desktop/SBLA/DylanSimulations/DLA_SBLA_z3','_015_vmax')
# t = hierarchical_sbla_maker(sblaprops_vmax_005,\
#                             '/home/dsantos/Desktop/SBLA/DylanSimulations/DLA_SBLA_z3','_005_vmax')
    
# t = hierarchical_sbla_maker(sblaprops_vdisp_025,\
#                             '/home/dsantos/Desktop/SBLA/DylanSimulations/DLA_SBLA_z3','_025_vdisp')
# t = hierarchical_sbla_maker(sblaprops_vdisp_015,\
#                             '/home/dsantos/Desktop/SBLA/DylanSimulations/DLA_SBLA_z3','_015_vdisp')
# t = hierarchical_sbla_maker(sblaprops_vdisp_005,\
#                             '/home/dsantos/Desktop/SBLA/DylanSimulations/DLA_SBLA_z3','_005_vdisp')
    
#%% Covering Fraction estimation
def cov_fraction(file, sbla_cgm):

    with fits.open(file) as hdul:
        sblas = hdul[1].data

    cov_fraction_all = []
    
    for k in tqdm(range(len(sblas))):
        cov_fraction = np.zeros(len(group_posx[mass_filter][sbla_cgm['Halo Index'][k]]))

        for i in range(len(group_posx[mass_filter][sbla_cgm['Halo Index'][k]])):
            # Total number of pixels inside the virial radius
            rad = np.where((f['ray_pos'][:,0] - group_posx[mass_filter][sbla_cgm['Halo Index'][k]][i])**2 +\
                           (f['ray_pos'][:,1] - group_posy[mass_filter][sbla_cgm['Halo Index'][k]][i])**2 <= group_vrad[mass_filter][sbla_cgm['Halo Index'][k]][i]**2)
            
            # Coordinates of SBLAs inside said virial radius and velocity scale
            coords = np.array([f['ray_pos'][:,0][sblas[k][0]][sbla_cgm['SBLAs Indices'][k][i]],f['ray_pos'][:,1][sblas[k][0]][sbla_cgm['SBLAs Indices'][k][i]]])
            
            counter = 0 # There can be more than one SBLA per line-of-sight, we need to remove them
            coord_tup = []
        
            for h in range(len(coords[0])): # Changing coordinates into tuples so there are individual items for each pair
                coord_tup.append(tuple(coords[:,h]))
            
            counted = Counter(coord_tup) # Counting how many repeated elements exist
            
            for h in counted.keys(): # Iterating over each element to see how many repeats there are
                counter += counted[h] - 1 # If counted of a specific key is 1, there is only one unique element, hence it should be 0 to not count as a repeat
            
            tot_pixels = len(rad[0]) # Gets total number of pixels from the numpy where
            cov = len(coords[0]) - counter # Ensures that we have one SBLA per line of sight
            
            cov_fraction[i] = cov/tot_pixels # Calculates the covering fraction and attaches it to array

        cov_fraction_allhalo = np.zeros(len(group_posx[mass_filter]))
        cov_fraction_allhalo[sbla_cgm['Halo Index'][k]] = cov_fraction

        cov_fraction_all.append(cov_fraction_allhalo)

    return cov_fraction_all

cov_fraction_vmax_025 = cov_fraction('DLA_SBLA_HighRes/SBLA_025_AllBins_Index.fits', sbla025_vmax_vrad)
cov_fraction_vmax_015 = cov_fraction('DLA_SBLA_HighRes/SBLA_015_AllBins_Index.fits', sbla015_vmax_vrad)
cov_fraction_vmax_005 = cov_fraction('DLA_SBLA_HighRes/SBLA_005_AllBins_Index.fits', sbla005_vmax_vrad)

cov_fraction_vdisp_025 = cov_fraction('DLA_SBLA_HighRes/SBLA_025_AllBins_Index.fits', sbla025_vdisp_vrad)
cov_fraction_vdisp_015 = cov_fraction('DLA_SBLA_HighRes/SBLA_015_AllBins_Index.fits', sbla015_vdisp_vrad)
cov_fraction_vdisp_005 = cov_fraction('DLA_SBLA_HighRes/SBLA_005_AllBins_Index.fits', sbla005_vdisp_vrad)

file = Table([cov_fraction_vmax_025])
file.write('DLA_SBLA_HighRes/CovFraction_vmax_025.fits',overwrite=True)
file = Table([cov_fraction_vmax_015])
file.write('DLA_SBLA_HighRes/CovFraction_vmax_015.fits',overwrite=True)
file = Table([cov_fraction_vmax_005])
file.write('DLA_SBLA_HighRes/CovFraction_vmax_005.fits',overwrite=True)

file = Table([cov_fraction_vdisp_025])
file.write('DLA_SBLA_HighRes/CovFraction_vdisp_025.fits',overwrite=True)
file = Table([cov_fraction_vdisp_015])
file.write('DLA_SBLA_HighRes/CovFraction_vdisp_015.fits',overwrite=True)
file = Table([cov_fraction_vdisp_005])
file.write('DLA_SBLA_HighRes/CovFraction_vdisp_005.fits',overwrite=True)