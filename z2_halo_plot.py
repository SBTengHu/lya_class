#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits, ascii
import h5py
from scipy.interpolate import interp2d, RegularGridInterpolator
from matplotlib.ticker import AutoMinorLocator,MaxNLocator
import astropy.table as tab

#cosmological parameters
c = 3e5  # km/s
lya = 1215.67  # AA
h = 0.6774
OmegaM = 0.3089
OmegaB = 0.0486
#starting redshift of the TNG50
z_0 = 2.0020281392528516
Lbox = 35  # Mpc/h
ncell = 1000#len(np.sort(np.unique(f['ray_pos'][:,0])))
cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=OmegaM, Ob0=OmegaB)
hubblez0 = cosmo.H(z_0)

#use the z in roughly the middle of the box
hubblez = cosmo.H(2.020)

dz0 = ((Lbox / h / ncell) * hubblez0 / c).value
dz = ((Lbox / h / ncell) * hubblez / c).value # dz in redshift per cell
dWL = (dz * lya)  # wl in AA per cell
dl = Lbox / ncell  # dist per cell for raw skewer in Mpc/h

#f = h5py.File('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/spectra_TNG50-1_z2.0_n2000d2-rndfullbox_SDSS-BOSS_HI_combined.hdf5', 'r')
f = h5py.File('/data/forest/dsantos/DylanSims/Data/z2/KECK/Random/spectra_TNG50-1_z2.0_n2000d2-rndfullbox_KECK-HIRES-B14_HI_combined.hdf5', 'r')
wavelength_all = np.array(f['wave'])

f1 = ascii.read('/data/forest/dsantos/DylanSims/Data/z2/SubhaloInfo.csv',format='csv')

subhalo_posx = np.array(f1['Subhalo_PosX']) # Subhalo positions in x
subhalo_posy = np.array(f1['Subhalo_PosY']) # Subhalo positions in y
subhalo_posz = np.array(f1['Subhalo_PosZ']) # Subhalo positions in z
subhalo_mass = np.array(f1['Subhalo_Mass']) # Subhalo mass. To convert to physical mass you have to multiply by 1e10/H0
subhalo_radhm = np.array(f1['Subhalo_HalfMassRadius']) # Subhalo half mass radius. Twice this is the virial radius.
subhalo_z = np.array(f1['Subhalo_GasMetal']) # Subhalo gas metallicity
subhalo_vz = np.array(f1['Subhalo_PVZ']) # Subhalo peculiar velocity in z axis (km/s)
subhalo_vdisp = np.array(f1['Subhalo_VDispersion']) # Subhalo velocity dispersion (km/s)
subhalo_vmax = np.array(f1['Subhalo_VMax']) # Subhalo maximum velocity of the rotation curve (km/s)

#f = ascii.read('/data/forest/dsantos/DylanSims/Data/z2/GroupInfo.csv',format='csv')
#group_posx = np.array(f['Group_CMX']) # Group positions in x
#group_posy = np.array(f['Group_CMY']) # Group positions in y
#group_posz = np.array(f['Group_CMZ']) # Group positions in z
#group_mass = np.array(f['Group_Mass']) # Group Mass
#group_z = np.array(f['Group_Metal']) # Group Metallicity
#group_vrad = np.array(f['Group_RCrit200']) # Group virial radius
#group_subhaloid = np.array(f['Subhalo_ID']) # Central Subhalo ID


# In[4]:


#DLA_list ='/data/forest/dsantos/DylanSims/Data/z2/KECK/Random/Old_SpecBins/DLA_SBLA_HighRes/DLA_AllBins_Index.fits'
DLA_list ='/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/DLA_AllBins_Index.fits'
f_DLA=tab.Table.read(DLA_list, hdu=1)

#mark DLA index
all_indices = np.arange(len(f['flux']))
not_DLA = np.isin(all_indices, f_DLA[0][0], invert=True)
ind_not_DLA = all_indices[not_DLA]
ind_DLA = all_indices[~not_DLA]


# In[5]:


#convert from z position to wavelength
z_halo = subhalo_posz/(dl*1000)  * dz+ (z_0)
#wl_halo = (1+z_halo )* lya
wl_halo = (1+z_halo + (1+z_halo)*subhalo_vz/c )* lya

#wavelength of the halos corresponding to their real space location. (vz=0)
wl_halo_v0 = (1+z_halo )* lya

# location of a mass=12 halo
mass_filter_0 = (np.log10(subhalo_mass*1e10/0.6774) > 11.45) & (np.log10(subhalo_mass*1e10/0.6774) < 11.55)
np.sum(mass_filter_0)


# In[19]:


halo_number=1
    
halo_0_x = subhalo_posx[mass_filter_0][halo_number]
wl_halo_0 = ( (z_0) + subhalo_posz[mass_filter_0][halo_number]/(dl*1000)  * dz
             + (1+z_halo[mass_filter_0][halo_number])*subhalo_vz[mass_filter_0][halo_number]/c +1 )* lya

halo_0_r = 2*subhalo_radhm[mass_filter_0][halo_number]

halo_0_dwl= (1+z_halo[mass_filter_0][halo_number])*subhalo_vmax[mass_filter_0][halo_number]/c * lya

#the width of slice in kpc/h
y_pos = subhalo_posy[mass_filter_0][halo_number]

dy=35

# extendthe bourandary a little bit 
wl_min = (1+z_0)*lya-5
wl_max = (1+z_0 + 1000*dz)*lya+5

y_ind = np.where((f['ray_pos'][:,1]>=y_pos-dy/2)&(f['ray_pos'][:,1]<=y_pos+dy/2))[0]
y_ind_not_DLA = np.intersect1d(y_ind, ind_not_DLA)

#label the DLA los in the panel
y_ind_DLA = np.intersect1d(y_ind, ind_DLA)
x_los_DLA = f['ray_pos'][y_ind_DLA][:,0]

wl_ind = np.where((wavelength_all<wl_max) & (wavelength_all>wl_min))[0]

ray_z = wavelength_all[wl_ind]

vel_arr_z = (ray_z-(ray_z[0]+ray_z[-1])/2)/wl_halo_0 * c # in km/s
d_arr_z = np.arange(len(vel_arr_z)) * dl *1000 # in kpc/h

#los that go through the halo
x_ind_halo0_los = \
np.where((f['ray_pos'][y_ind_not_DLA][:, 0] >= halo_0_x - halo_0_r) & (f['ray_pos'][y_ind_not_DLA][:, 0] <= halo_0_x + halo_0_r))[0]
x_halo0_los = f['ray_pos'][y_ind_not_DLA][:, 0][x_ind_halo0_los]

wl_halo_0_max = wl_halo_0 + (1+z_0+subhalo_posz[mass_filter_0][halo_number]/(dl*1000) * dz)*subhalo_vmax[mass_filter_0][halo_number]/c * lya
wl_halo_0_min = wl_halo_0 - (1+z_0+subhalo_posz[mass_filter_0][halo_number]/(dl*1000) * dz)*subhalo_vmax[mass_filter_0][halo_number]/c * lya

#condition if for all subhalos in the plane of the slice
#condition0 = (subhalo_posy - y_pos)**2  <= (35/2)**2
condition1 = (subhalo_posy - y_pos)**2  <= (2*subhalo_radhm)**2
condition2 = (np.log10(subhalo_mass*1e10/0.6774) > 9.05)

cond_all = condition1 & condition2

#the radius projected in x-axis
subhalo_rx= np.zeros(len(subhalo_posx[cond_all]))
subhalo_rx = np.sqrt(4*subhalo_radhm[cond_all]**2 - (subhalo_posy[cond_all] - y_pos)**2)
subhalo_dwl = ((1+z_halo[cond_all])*subhalo_vmax[cond_all]/c )* lya
#label all pixels in the halo range

from scipy.spatial import cKDTree

# Assuming f, subhalo_posx, subhalo_posy, subhalo_radhm, subhalo_vz, subhalo_vmax, z_halo, lya, c, and ray_z are already defined
LOS_xy = np.array((f['ray_pos'][:, 0], f['ray_pos'][:, 1])).T
subhalo_positions = np.array((subhalo_posx[condition2], subhalo_posy[condition2])).T
subhalo_radii = 2 * subhalo_radhm[condition2]
z_halo_cond = z_halo[condition2]
subhalo_vz_cond =subhalo_vz[condition2]
subhalo_vmax_cond = subhalo_vmax[condition2]
 
# Create a KDTree for LOS_xy
tree = cKDTree(LOS_xy)

# Initialize lists to store results
los_halo_ind = []
wl_halo_ind_all = []

# Create the LOS_MASK array
LOS_MASK = np.zeros((len(f['ray_pos']), len(ray_z)), dtype=np.float16)

# Iterate over each halo
for i, (pos, radius) in enumerate(zip(subhalo_positions, subhalo_radii)):
    # Find all LOS within the radius in the x-y plane
    los_indices0 = tree.query_ball_point(pos, radius)

    # Store the results
    los_halo_ind.append(los_indices0)

    # Calculate the wavelength range for the halo in the z-direction
    c_z_i = (z_halo_cond[i]+1)
    halo_dwl_min = (1 + z_halo_cond[i] + c_z_i*subhalo_vz_cond[i] / c - c_z_i*subhalo_vmax_cond[i] / c) * lya
    halo_dwl_max = (1 + z_halo_cond[i] + c_z_i*subhalo_vz_cond[i] / c + c_z_i*subhalo_vmax_cond[i] / c) * lya
    
    # Find the wavelength indices that are covered by the halo
    wl_indices0 = np.where((ray_z >= halo_dwl_min) & (ray_z <= halo_dwl_max))[0]
    
    #if i <= 30:
    #    print(z_halo_cond[i],halo_dwl_min,halo_dwl_max,2*subhalo_radhm[condition2][i],len(wl_indices0), len(los_indices0),los_indices0[0:10])

    # the halo wl for each los that intersect with halo i     
    wl_los_halo_indices = []
    for j in los_indices0:
        # Store the results
        wl_los_halo_indices.append(wl_indices0)
        LOS_MASK[j, wl_indices0] = 1

    wl_halo_ind_all.append(wl_los_halo_indices)

# los_indices now contains the indices of LOS in the x-y plane for each halo
# wavelength_indices contains the wavelength indices covered by each halo in the z-direction
# only use the los around the halo
#the plot range in unit of virial radius
r_f = 3

#los that go through the halo
x_ind_plot= \
np.where((f['ray_pos'][y_ind_not_DLA][:, 0] >= halo_0_x - r_f*halo_0_r) & (f['ray_pos'][y_ind_not_DLA][:, 0] <= halo_0_x + r_f*halo_0_r))[0]

#flux in wl range
#flux_lya = f['flux'][:, wl_ind]
#flux_map= flux_lya[y_ind_not_DLA]

flux_map = f['flux'][y_ind_not_DLA[x_ind_plot]][:, wl_ind]

#remove los with repeated x
ind_unique = np.unique(f['ray_pos'][:,0][y_ind_not_DLA[x_ind_plot]], return_index=True)
x_pos_unique = f['ray_pos'][:,0][y_ind_not_DLA[x_ind_plot]][ind_unique[1]]

flux_map_unique = flux_map[ind_unique[1]]

ind_sort_unique = np.argsort(x_pos_unique)
x_unique_sort = x_pos_unique[ind_sort_unique]

flux_map_unique_sort = flux_map_unique[ind_sort_unique]
map0=np.array(np.vstack(flux_map_unique))

#interpolate the color map to finer grid
f_map = RegularGridInterpolator((x_unique_sort, ray_z), map0, method='cubic')

x_grid = np.linspace(x_unique_sort[0],x_unique_sort[-1],10*ncell)
X_plot, Y_plot = np.meshgrid(x_grid[0:-1], ray_z[0:-1], indexing='ij')

map_plot = f_map((X_plot,Y_plot))


# In[20]:


gridspec = {'width_ratios': [1,0.07,1, 0.07]}
fig3, axes3 = plt.subplots(nrows=1, ncols=4, figsize=(8,8),gridspec_kw=gridspec)

ax3 = axes3[0].twinx()
c2 = axes3[0].pcolormesh(x_grid,ray_z , map_plot.T, cmap='Greys', vmin=0., vmax=1.)

#axes2[0].scatter(subhalo_posx[cond_all],wl_halo[cond_all], marker='*', s=2, c='red')
axes3[0].errorbar(subhalo_posx[cond_all],wl_halo[cond_all],
                  xerr=subhalo_rx,yerr=subhalo_dwl, fmt='.', linewidth=1, capsize=1, c='red')

for x_i in x_los_DLA:
    axes3[0].axvline(x=x_i, ls=':', color='green', lw=1.0, alpha=0.5,label = 'DLA')

for x_i in x_halo0_los:
    axes3[0].axvline(x=x_i, ls='-', color='yellow', lw=1.0, alpha=0.5,label = 'LOS')


ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()

#ax3.set_yticks(d_arr_z)

#plot mass=12 halo
#axes2[0].scatter(np.atleast_1d(halo_0_x),np.atleast_1d(wl_halo_0_wl), marker='*', s=1, c='green')

axes3[0].set_title('flux')
# set the limits of the plot to the limits of the data
plt.colorbar(c2, cax=axes3[1])

axes3[0].set_ylabel(r'wavelength',fontsize=14)
axes3[0].set_xlabel(r'X [ckpc/h]',fontsize=14)
axes3[0].set_xlim(halo_0_x-2*halo_0_r, halo_0_x+2*halo_0_r)
axes3[0].set_ylim(wl_halo_0-2*halo_0_dwl,np.min([ray_z[-1],wl_halo_0+2*halo_0_dwl]))
#axes3[0].set_ylim(ray_z[0],ray_z[-1])

axes3[0].xaxis.set_minor_locator(AutoMinorLocator())
axes3[0].yaxis.set_minor_locator(AutoMinorLocator())
#embed()

c2 = axes3[2].pcolormesh(x_grid,ray_z , map_plot.T, cmap='Blues', vmin=0., vmax=0.25)

#axes2[0].scatter(subhalo_posx[cond_all],wl_halo[cond_all], marker='*', s=2, c='red')
axes3[2].errorbar(subhalo_posx[cond_all],wl_halo[cond_all],
                  xerr=subhalo_rx,yerr=subhalo_dwl, fmt='.', linewidth=1, capsize=1, c='red')

for x_i in x_los_DLA:
    axes3[2].axvline(x=x_i, ls=':', color='green', lw=1.0, alpha=0.5,label = 'DLA')

for x_i in x_halo0_los:
    axes3[2].axvline(x=x_i, ls='-', color='yellow', lw=1.0, alpha=0.5,label = 'LOS')

#plot mass=12 halo
#axes2[0].scatter(np.atleast_1d(halo_0_x),np.atleast_1d(wl_halo_0_wl), marker='*', s=1, c='green')

axes3[2].set_title('flux')
# set the limits of the plot to the limits of the data
plt.colorbar(c2, cax=axes3[3])

axes3[2].set_ylabel(r'wavelength',fontsize=14)
axes3[2].set_xlabel(r'X [ckpc/h]',fontsize=14)
axes3[2].set_xlim(halo_0_x-4*halo_0_r, halo_0_x+4*halo_0_r)
axes3[2].set_ylim(wl_halo_0-2*halo_0_dwl,np.min([ray_z[-1],wl_halo_0+2*halo_0_dwl]))
#axes3[0].set_ylim(ray_z[0],ray_z[-1])

axes3[2].xaxis.set_minor_locator(AutoMinorLocator())
axes3[2].yaxis.set_minor_locator(AutoMinorLocator())
#embed()
#fig3.legend(loc='upper right')
#fig3.tight_layout()
plt.subplots_adjust(wspace=1.5, hspace=0)
plt.show()
#savename2 =('z2_zoomin_halos_y'+str(i_halo)+'_mass11.5.pdf')
#plt.savefig(savename2, dpi=100)
plt.close()


# In[17]:


for i_halo in np.arange(1,2): #np.sum(mass_filter_0)):
    print("halo ",i_halo)
    
    halo_number=i_halo
        
    halo_0_x = subhalo_posx[mass_filter_0][halo_number]
    wl_halo_0 = ( (z_0) + subhalo_posz[mass_filter_0][halo_number]/(dl*1000)  * dz
                 + (1+z_halo[mass_filter_0][halo_number])*subhalo_vz[mass_filter_0][halo_number]/c +1 )* lya
    
    halo_0_r = 2*subhalo_radhm[mass_filter_0][halo_number]
    
    halo_0_dwl= (1+z_halo[mass_filter_0][halo_number])*subhalo_vmax[mass_filter_0][halo_number]/c * lya
    
    #the width of slice in kpc/h
    y_pos = subhalo_posy[mass_filter_0][halo_number]
    
    dy=35
    
    # extendthe bourandary a little bit 
    wl_min = (1+z_0)*lya-5
    wl_max = (1+z_0 + 1000*dz)*lya+5
    
    y_ind = np.where((f['ray_pos'][:,1]>=y_pos-dy/2)&(f['ray_pos'][:,1]<=y_pos+dy/2))[0]
    y_ind_not_DLA = np.intersect1d(y_ind, ind_not_DLA)
    
    #label the DLA los in the panel
    y_ind_DLA = np.intersect1d(y_ind, ind_DLA)
    x_los_DLA = f['ray_pos'][y_ind_DLA][:,0]
    
    wl_ind = np.where((wavelength_all<wl_max) & (wavelength_all>wl_min))[0]
    
    ray_z = wavelength_all[wl_ind]
    
    vel_arr_z = (ray_z-(ray_z[0]+ray_z[-1])/2)/wl_halo_0 * c # in km/s
    d_arr_z = np.arange(len(vel_arr_z)) * dl *1000 # in kpc/h
    
    #los that go through the halo
    x_ind_halo0_los = \
    np.where((f['ray_pos'][y_ind_not_DLA][:, 0] >= halo_0_x - halo_0_r) & (f['ray_pos'][y_ind_not_DLA][:, 0] <= halo_0_x + halo_0_r))[0]
    x_halo0_los = f['ray_pos'][y_ind_not_DLA][:, 0][x_ind_halo0_los]
    
    wl_halo_0_max = wl_halo_0 + (1+z_0+subhalo_posz[mass_filter_0][halo_number]/(dl*1000) * dz)*subhalo_vmax[mass_filter_0][halo_number]/c * lya
    wl_halo_0_min = wl_halo_0 - (1+z_0+subhalo_posz[mass_filter_0][halo_number]/(dl*1000) * dz)*subhalo_vmax[mass_filter_0][halo_number]/c * lya
    
    #condition if for all subhalos in the plane of the slice
    #condition0 = (subhalo_posy - y_pos)**2  <= (35/2)**2
    condition1 = (subhalo_posy - y_pos)**2  <= (2*subhalo_radhm)**2
    condition2 = (np.log10(subhalo_mass*1e10/0.6774) > 9.05)
    
    cond_all = condition1 & condition2
    
    #the radius projected in x-axis
    subhalo_rx= np.zeros(len(subhalo_posx[cond_all]))
    subhalo_rx = np.sqrt(4*subhalo_radhm[cond_all]**2 - (subhalo_posy[cond_all] - y_pos)**2)
    subhalo_dwl = ((1+z_halo[cond_all])*subhalo_vmax[cond_all]/c )* lya
    #label all pixels in the halo range
    
    from scipy.spatial import cKDTree
    
    # Assuming f, subhalo_posx, subhalo_posy, subhalo_radhm, subhalo_vz, subhalo_vmax, z_halo, lya, c, and ray_z are already defined
    LOS_xy = np.array((f['ray_pos'][:, 0], f['ray_pos'][:, 1])).T
    subhalo_positions = np.array((subhalo_posx[condition2], subhalo_posy[condition2])).T
    subhalo_radii = 2 * subhalo_radhm[condition2]
    z_halo_cond = z_halo[condition2]
    subhalo_vz_cond =subhalo_vz[condition2]
    subhalo_vmax_cond = subhalo_vmax[condition2]
     
    # Create a KDTree for LOS_xy
    tree = cKDTree(LOS_xy)
    
    # Initialize lists to store results
    los_halo_ind = []
    wl_halo_ind_all = []
    
    # Create the LOS_MASK array
    LOS_MASK = np.zeros((len(f['ray_pos']), len(ray_z)), dtype=np.float16)
    
    # Iterate over each halo
    for i, (pos, radius) in enumerate(zip(subhalo_positions, subhalo_radii)):
        # Find all LOS within the radius in the x-y plane
        los_indices0 = tree.query_ball_point(pos, radius)
    
        # Store the results
        los_halo_ind.append(los_indices0)
    
        # Calculate the wavelength range for the halo in the z-direction
        c_z_i = (z_halo_cond[i]+1)
        halo_dwl_min = (1 + z_halo_cond[i] + c_z_i*subhalo_vz_cond[i] / c - c_z_i*subhalo_vmax_cond[i] / c) * lya
        halo_dwl_max = (1 + z_halo_cond[i] + c_z_i*subhalo_vz_cond[i] / c + c_z_i*subhalo_vmax_cond[i] / c) * lya
        
        # Find the wavelength indices that are covered by the halo
        wl_indices0 = np.where((ray_z >= halo_dwl_min) & (ray_z <= halo_dwl_max))[0]
        
        #if i <= 30:
        #    print(z_halo_cond[i],halo_dwl_min,halo_dwl_max,2*subhalo_radhm[condition2][i],len(wl_indices0), len(los_indices0),los_indices0[0:10])
    
        # the halo wl for each los that intersect with halo i     
        wl_los_halo_indices = []
        for j in los_indices0:
            # Store the results
            wl_los_halo_indices.append(wl_indices0)
            LOS_MASK[j, wl_indices0] = 1
    
        wl_halo_ind_all.append(wl_los_halo_indices)
    
    # los_indices now contains the indices of LOS in the x-y plane for each halo
    # wavelength_indices contains the wavelength indices covered by each halo in the z-direction
    # only use the los around the halo
    #the plot range in unit of virial radius
    r_f = 3
    
    #los that go through the halo
    x_ind_plot= \
    np.where((f['ray_pos'][y_ind_not_DLA][:, 0] >= halo_0_x - r_f*halo_0_r) & (f['ray_pos'][y_ind_not_DLA][:, 0] <= halo_0_x + r_f*halo_0_r))[0]

    #flux in wl range
    #flux_lya = f['flux'][:, wl_ind]
    #flux_map= flux_lya[y_ind_not_DLA]
    
    flux_map = f['flux'][y_ind_not_DLA[x_ind_plot]][:, wl_ind]

    #remove los with repeated x
    ind_unique = np.unique(f['ray_pos'][:,0][y_ind_not_DLA[x_ind_plot]], return_index=True)
    x_pos_unique = f['ray_pos'][:,0][y_ind_not_DLA[x_ind_plot]][ind_unique[1]]
    
    flux_map_unique = flux_map[ind_unique[1]]
    
    ind_sort_unique = np.argsort(x_pos_unique)
    x_unique_sort = x_pos_unique[ind_sort_unique]

    #only plot the halos with more than 10 los
    Mini_LOS = 10
    if len(x_unique_sort) <= Mini_LOS:
        print("n(LOS) <= 10, skip")
        continue
    
    flux_map_unique_sort = flux_map_unique[ind_sort_unique]
    map0=np.array(np.vstack(flux_map_unique))

    #interpolate the color map to finer grid
    f_map = RegularGridInterpolator((x_unique_sort, ray_z), map0, method='cubic')
    
    x_grid = np.linspace(x_unique_sort[0],x_unique_sort[-1],10*ncell)
    X_plot, Y_plot = np.meshgrid(x_grid[0:-1], ray_z[0:-1], indexing='ij')
    
    map_plot = f_map((X_plot,Y_plot))
    
    gridspec = {'width_ratios': [1,0.07,1, 0.07]}
    fig3, axes3 = plt.subplots(nrows=1, ncols=4, figsize=(8,8),gridspec_kw=gridspec)
    
    c2 = axes3[0].pcolormesh(x_grid,ray_z , map_plot.T, cmap='Greys', vmin=0., vmax=1.)
    
    #axes2[0].scatter(subhalo_posx[cond_all],wl_halo[cond_all], marker='*', s=2, c='red')
    axes3[0].errorbar(subhalo_posx[cond_all],wl_halo[cond_all],
                      xerr=subhalo_rx,yerr=subhalo_dwl, fmt='.', linewidth=1, capsize=1, c='red')
    
    for x_i in x_los_DLA:
        axes3[0].axvline(x=x_i, ls=':', color='green', lw=1.0, alpha=0.5,label = 'DLA')
    
    for x_i in x_halo0_los:
        axes3[0].axvline(x=x_i, ls='-', color='yellow', lw=1.0, alpha=0.5,label = 'LOS')

    ax3 = axes3[0].twinx()
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.set_yticks(d_arr_z)
    
    #plot mass=12 halo
    #axes2[0].scatter(np.atleast_1d(halo_0_x),np.atleast_1d(wl_halo_0_wl), marker='*', s=1, c='green')
    
    axes3[0].set_title('flux')
    # set the limits of the plot to the limits of the data
    plt.colorbar(c2, cax=axes3[1])
    
    axes3[0].set_ylabel(r'wavelength',fontsize=14)
    axes3[0].set_xlabel(r'X [ckpc/h]',fontsize=14)
    axes3[0].set_xlim(halo_0_x-2*halo_0_r, halo_0_x+2*halo_0_r)
    axes3[0].set_ylim(wl_halo_0-2*halo_0_dwl,np.min([ray_z[-1],wl_halo_0+2*halo_0_dwl]))
    #axes3[0].set_ylim(ray_z[0],ray_z[-1])
    
    axes3[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes3[0].yaxis.set_minor_locator(AutoMinorLocator())
    #embed()
    
    c2 = axes3[2].pcolormesh(x_grid,ray_z , map_plot.T, cmap='Blues', vmin=0., vmax=0.25)
    
    #axes2[0].scatter(subhalo_posx[cond_all],wl_halo[cond_all], marker='*', s=2, c='red')
    axes3[2].errorbar(subhalo_posx[cond_all],wl_halo[cond_all],
                      xerr=subhalo_rx,yerr=subhalo_dwl, fmt='.', linewidth=1, capsize=1, c='red')
    
    for x_i in x_los_DLA:
        axes3[2].axvline(x=x_i, ls=':', color='green', lw=1.0, alpha=0.5,label = 'DLA')
    
    for x_i in x_halo0_los:
        axes3[2].axvline(x=x_i, ls='-', color='yellow', lw=1.0, alpha=0.5,label = 'LOS')
    
    #plot mass=12 halo
    #axes2[0].scatter(np.atleast_1d(halo_0_x),np.atleast_1d(wl_halo_0_wl), marker='*', s=1, c='green')
    
    axes3[2].set_title('flux')
    # set the limits of the plot to the limits of the data
    plt.colorbar(c2, cax=axes3[3])
    
    axes3[2].set_ylabel(r'wavelength',fontsize=14)
    axes3[2].set_xlabel(r'X [ckpc/h]',fontsize=14)
    axes3[2].set_xlim(halo_0_x-4*halo_0_r, halo_0_x+4*halo_0_r)
    axes3[2].set_ylim(wl_halo_0-2*halo_0_dwl,np.min([ray_z[-1],wl_halo_0+2*halo_0_dwl]))
    #axes3[0].set_ylim(ray_z[0],ray_z[-1])
    
    axes3[2].xaxis.set_minor_locator(AutoMinorLocator())
    axes3[2].yaxis.set_minor_locator(AutoMinorLocator())
    #embed()
    #fig3.legend(loc='upper right')
    #fig3.tight_layout()
    plt.subplots_adjust(wspace=1.5, hspace=0)
    plt.show()
    #savename2 =('z2_zoomin_halos_y'+str(i_halo)+'_mass11.5.pdf')
    #plt.savefig(savename2, dpi=100)
    plt.close()


# In[ ]:




