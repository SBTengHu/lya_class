#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits, ascii
import h5py
from scipy.interpolate import interp2d, RegularGridInterpolator
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import astropy.table as tab
from sys import getsizeof
import os
import glob
from scipy.spatial import cKDTree
import time

# cosmological parameters
c = 3e5  # km/s
lya = 1215.67  # AA
h = 0.6774
OmegaM = 0.3089
OmegaB = 0.0486
# starting redshift of the TNG50
z_0 = 2.0020281392528516
Lbox = 35  # Mpc/h
ncell = 1000  # len(np.sort(np.unique(f['ray_pos'][:,0])))
cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=OmegaM, Ob0=OmegaB)
# hubblez = cosmo.H(z_0)
hubblez = cosmo.H(2.02)

dz = ((Lbox / h / ncell) * hubblez / c).value  # dz in redshift per cell
dWL = (dz * lya)  # wl in AA per cell
dl = Lbox / ncell  # dist per cell for raw skewer in Mpc/h

f = h5py.File(
    '/data/forest/dsantos/DylanSims/Data/z2/KECK/Random/spectra_TNG50-1_z2.0_n2000d2-rndfullbox_KECK-HIRES-B14_HI_combined.hdf5','r')
# f = h5py.File('/data/forest/dsantos/DylanSims/Data/z2/KECK/Uniform/
#spectra_TNG50-1_z2.0_n1000d2-fullbox_KECK-HIRES-B14_HI_combined.hdf5', 'r')
wavelength_all = np.array(f['wave'])

f1 = ascii.read('/data/forest/dsantos/DylanSims/Data/z2/SubhaloInfo.csv', format='csv')

subhalo_posx = np.array(f1['Subhalo_PosX'])  # Subhalo positions in x
subhalo_posy = np.array(f1['Subhalo_PosY'])  # Subhalo positions in y
subhalo_posz = np.array(f1['Subhalo_PosZ'])  # Subhalo positions in z
subhalo_mass = np.array(f1['Subhalo_Mass'])  # Subhalo mass. To convert to physical mass you have to multiply by 1e10/H0
subhalo_radhm = np.array(f1['Subhalo_HalfMassRadius'])  # Subhalo half mass radius. Twice this is the virial radius.
subhalo_z = np.array(f1['Subhalo_GasMetal'])  # Subhalo gas metallicity
subhalo_vz = np.array(f1['Subhalo_PVZ'])  # Subhalo peculiar velocity in z axis (km/s)
subhalo_vdisp = np.array(f1['Subhalo_VDispersion'])  # Subhalo velocity dispersion (km/s)
subhalo_vmax = np.array(f1['Subhalo_VMax'])  # Subhalo maximum velocity of the rotation curve (km/s)

# f = ascii.read('/data/forest/dsantos/DylanSims/Data/z2/GroupInfo.csv',format='csv')
# group_posx = np.array(f['Group_CMX']) # Group positions in x
# group_posy = np.array(f['Group_CMY']) # Group positions in y
# group_posz = np.array(f['Group_CMZ']) # Group positions in z
# group_mass = np.array(f['Group_Mass']) # Group Mass
# group_z = np.array(f['Group_Metal']) # Group Metallicity
# group_vrad = np.array(f['Group_RCrit200']) # Group virial radius
# group_subhaloid = np.array(f['Subhalo_ID']) # Central Subhalo ID

# Assuming subhalo_mass is already defined
# Multiply subhalo_mass by the factor
subhalo_mass_corrected = subhalo_mass * 1e10 / 0.6774

# Filter halos with mass higher than 1e9
mass_threshold = 1e9
filtered_subhalo_mass = subhalo_mass_corrected[subhalo_mass_corrected > mass_threshold]

# Calculate the logarithm of the filtered masses
log_filtered_subhalo_mass = np.log10(filtered_subhalo_mass)


# In[2]:


mass_bins=np.linspace(10, 11.5, 4)
bin_edges = mass_bins
#np.arange(np.floor(log_filtered_subhalo_mass.min()), np.ceil(log_filtered_subhalo_mass.max()) + 0.5, 0.5)

# Create a figure and axes for plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Draw a histogram for the logarithm of the filtered subhalo_mass
ax.hist(log_filtered_subhalo_mass, bins=bin_edges, color='blue', alpha=0.7, histtype='bar', rwidth=0.95)
ax.set_ylabel('Log10 Subhalo Mass [$\log_{10}(M_\odot)$]')
ax.set_ylabel('Halos Counts')

ax.set_yscale('log')
ax.set_ylim(1,2e5)
ax.grid(True)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

plt.show()


# In[3]:


DLA_list = '/data/forest/dsantos/DylanSims/Data/z2/KECK/Random/Old_SpecBins/DLA_SBLA_HighRes/DLA_AllBins_Index.fits'
f_DLA = tab.Table.read(DLA_list, hdu=1)

# Mark DLA index
all_indices = np.arange(len(f['flux']))
# Directly use the indices from f_DLA for marking DLAs
is_DLA = np.zeros(len(f['flux']), dtype=bool)
is_DLA[f_DLA[0][0]] = True
not_DLA_ind = np.where(~is_DLA)[0]

# Efficiently access and transform only the necessary data from 'ray_pos'
LOS_xy = np.vstack((f['ray_pos'][:, 0], f['ray_pos'][:, 1])).T

# convert from z position to wavelength
z_halo = subhalo_posz / (dl * 1000) * dz + (z_0)
# wl_halo = (1+z_halo )* lya
wl_halo = (1 + z_halo + (1+z_halo)*subhalo_vz / c) * lya

# extendthe bourandary a little bit
wl_min = (1 + z_0) * lya - 5
wl_max = (1 + z_0 + 1000 * dz) * lya + 5

wl_ind = np.where((wavelength_all < wl_max) & (wavelength_all > wl_min))[0]

ray_z = wavelength_all[wl_ind]


# In[4]:


# Initialize lists to store results
los_halo_ind = []
wl_halo_ind_all = []
#flattened_los_halo_ind = np.array([])
los_halo_bins = []

start_time = time.time()

# Create the LOS_MASK array
LOS_MASK = np.zeros((len(f['ray_pos']), len(ray_z)), dtype=np.float16)
print(f"Runtime: {time.time() - start_time:.2f}")

# Create a KDTree for LOS_xy
tree = cKDTree(LOS_xy)
print(f"Runtime: {time.time() - start_time:.2f}")


# In[ ]:


for ibin in np.arange(0,len(mass_bins)-1):

    start_time = time.time()
    
    mass_cond = ((np.log10(subhalo_mass * 1e10 / 0.6774) > mass_bins[ibin])
                 & (np.log10(subhalo_mass * 1e10 / 0.6774) < mass_bins[ibin+1]))

    subhalo_positions = np.array((subhalo_posx[mass_cond], subhalo_posy[mass_cond])).T
    subhalo_radii = 2 * subhalo_radhm[mass_cond]
    z_halo_cond = z_halo[mass_cond]
    subhalo_vz_cond = (1+z_halo_cond)*subhalo_vz[mass_cond]
    subhalo_vmax_cond = subhalo_vmax[mass_cond]


    print(len(subhalo_positions),np.int16(ibin),mass_bins[ibin],mass_bins[ibin+1])

    los_halo_ibin = np.array([])

    # Iterate over each halo# Iterate over each halo
    for j, (pos, radius) in enumerate(zip(subhalo_positions, subhalo_radii)):
        
        # Find all LOS within the radius in the x-y plane
        los_indices_j = tree.query_ball_point(pos, radius)

        # Store the results
        los_halo_ind.append(los_indices_j)
        #flattened_los_halo_ind = np.concatenate(los_halo_ind)
        los_halo_ibin = np.concatenate(los_halo_ind)
        
        c_z_j = 1 + z_halo_cond[j]
        # Calculate the wavelength range for the halo in the z-direction
        halo_dwl_min = c_z_j* (1  +  subhalo_vz_cond[j]/c - subhalo_vmax_cond[j]/c) * lya
        
        halo_dwl_max = c_z_j* (1  +  subhalo_vz_cond[j]/c + subhalo_vmax_cond[j]/c) * lya

        # Find the wavelength indices that are covered by the halo
        wl_indices_j = np.where((ray_z >= halo_dwl_min) & (ray_z <= halo_dwl_max))[0]

        # if i <= 30:
        #   print(z_halo_cond[i],halo_dwl_min,halo_dwl_max,2*subhalo_radhm[condition2][i],len(wl_indices0), len(los_indices0),los_indices0[0:10])

        # the halo wl for each los that intersect with halo i
        wl_los_halo_indices = []
        for k in los_indices_j:
            # Store the results
            wl_los_halo_indices.append(wl_indices_j)
            LOS_MASK[k, wl_indices_j] = (mass_bins[ibin]+mass_bins[ibin+1])/2

        wl_halo_ind_all.append(wl_los_halo_indices)

        if not j%1000 :
            print(j,f"Runtime: {time.time() - start_time:.2f}")
            
    #a 2d list, each element/array correponds to all los that intersect with halo within a manss range
    los_halo_bins.append(los_halo_ibin)

    # Calculate and print the runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.2f}")


# In[22]:


#los with halos but including DLA
#LOSwHalo_ind = np.int32(np.unique(flattened_los_halo_ind))
halo_los_all = np.array(np.unique(np.concatenate(los_halo_bins)), dtype=np.int32)

#number of los with halos per halo mass bin
N_los_halo_bin = 1000
N_los_NO_halo = 5000


# In[49]:


los_halo_bins


# In[9]:


flux_halo_binsAll= []
mask_halo_binsAll= []
#start_time = time.time()
for i in np.arange(0,len(los_halo_bins)):
    #print(0,f"Runtime: {time.time() - start_time:.2f}")
    
    LOSHalo_notDLA_ibin =  np.array(np.intersect1d(not_DLA_ind, los_halo_bins[i]), dtype=np.int32)
    #print(len(LOSHalo_notDLA_ibin),1,f"Runtime: {time.time() - start_time:.2f}")
    
    flux_halo_bin = f['flux'][LOSHalo_notDLA_ibin[:N_los_halo_bin]][:, wl_ind]
    mask_halo_bin = LOS_MASK[LOSHalo_notDLA_ibin[:N_los_halo_bin]]
    #print(2,f"Runtime: {time.time() - start_time:.2f}")
    
    flux_halo_binsAll.append(flux_halo_bin)
    mask_halo_binsAll.append(mask_halo_bin)
    #print(3,f"Runtime: {time.time() - start_time:.2f}")


# In[37]:


# Find indices in not_DLA_ind that are not in flattened_los_halo_ind
LOSHalo_notDLA_ind = np.intersect1d(not_DLA_ind, halo_los_all)
LOSnoHalo_notDLA_ind = np.setdiff1d(not_DLA_ind, halo_los_all)


# In[40]:


mask_NO_halo = LOS_MASK[LOSnoHalo_notDLA_ind[:N_los_NO_halo]]
flux_NO_halo = f['flux'][LOSnoHalo_notDLA_ind[:N_los_NO_halo]][:, wl_ind]


# In[ ]:


version = '3label_test'
# Save flux_subset and LOS_MASK to an HDF5 file
with h5py.File('ML_training_z'+str(z_0)[0:3]+f'_spec{N_los_halo:d}_{N_los_NO_halo:d}'+version+'.hdf5', 'w') as hf:
    hf.create_dataset('flux_halo', data=flux_halo_binsAll)
    hf.create_dataset('flux_NO_halo', data=flux_NO_halo)
    hf.create_dataset('mask_halo', data=mask_halo_binsAll)
    hf.create_dataset('mask_NO_halo', data=mask_NO_halo)
    hf.create_dataset('wavelength_range', data=ray_z)


# In[48]:


random_indices


# In[47]:


version = '3label_test'
plotN=20
random_indices = np.random.choice(np.arange(len(flux_halo_binsAll[0])), plotN, replace=False)

for x in np.arange(0,plotN):
    flux = flux_halo_binsAll[0][random_indices][x][wl_ind]
    wave = ray_z
    mask = mask_halo_binsAll[0][random_indices][x]

    
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.plot(wave, flux,
            color='0.5', drawstyle='steps', zorder=1, lw=1.0)

    #ax.axvspan(wl_halo_0_min, wl_halo_0_max, alpha=0.5, color='yellow')

#    ax.annotate(f'(x,y): ({LOS_xy[x][0]:.2f}, {LOS_xy[x][1]:.2f})', xy=(0.15, 0.90), xycoords='axes fraction',
#                fontsize=10, color='black', ha='left', va='top')

    # Adding Twin Axes to plot using dataset_2
    ax2 = ax.twinx() 
     
    color = 'tab:green'
    ax2.tick_params(axis ='y', labelcolor = color) 

    ax2.plot(wave, mask,
            color='red', drawstyle='steps', ls='--', alpha=0.75, zorder=1, lw=1.0)
    
    ax2.set_ylabel('Halo Mass', color = color) 

    ax.set_ylabel('Transmission')
    ax.set_xlabel('$\\lambda / \\AA$')

    ax.set_ylim(-0.1, 1.25)
    ax2.set_ylim(-0.5, 13.5)
    ax.set_xlim(ray_z[0], ray_z[-1])

    plt.savefig('ML_spec' + str(x) + '_plot'+version+'.pdf')
    plt.close()


fitpdf = sorted([os.path.basename(x) for x in glob.glob('ML_spec*'+version+'.pdf')])
from PyPDF2 import PdfMerger
merger = PdfMerger()
for pdf in fitpdf:
    merger.append(pdf)
    os.remove(pdf)
merger.write(f'plot_SPEC_ML_N{plotN:d}_'+version+'.pdf')
merger.close()


# In[ ]:



