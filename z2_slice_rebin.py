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
from matplotlib.ticker import AutoMinorLocator,MaxNLocator
import astropy.table as tab
import spectres

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


# In[2]:


def wave_rebin(bins, wavelength):
    wavelength_rebin = np.zeros(int(len(wavelength)/bins))

    i = 0
    wi = 0
    while wi < len(wavelength_rebin):
        wavelength_rebin[wi] = np.sum(wavelength[i:i+bins])/bins
        wi += 1
        i += bins

    lya = np.where((wavelength_rebin > 3640) & (wavelength_rebin < 3652)) # Lyman alpha at z=2. is 3647.01 Angstroms  
    
    wave1 = wavelength_rebin[lya][-1]
    wave2 = wavelength_rebin[lya][-2]
    
    c = 2.99792e5 #km/s
    wave_lya = 1215.670 #Angstrom
    z1 = wave1/wave_lya - 1
    z2 = wave2/wave_lya - 1
    dz = (wave1 - wave2)/wave_lya
    deltav = c * dz/(1+z1)
    
    return wavelength_rebin, deltav


# In[3]:


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


len(ind_not_DLA),len(ind_DLA)


# In[6]:


len(ind_not_DLA)/4000000


# In[7]:


#convert from z position to wavelength
z_halo = subhalo_posz/(dl*1000)  * dz+ (z_0)
#wl_halo = (1+z_halo )* lya
wl_halo = (1+z_halo + (1+z_halo)*subhalo_vz/c )* lya

#wavelength of the halos corresponding to their real space location. (vz=0)
wl_halo_v0 = (1+z_halo )* lya

# location of a mass=12 halo
mass_filter_0 = (np.log10(subhalo_mass*1e10/0.6774) > 13.0) & (np.log10(subhalo_mass*1e10/0.6774) < 13.25)

#the ith halo
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
subhalo_rx= np.zeros(len(subhalo_posx))
subhalo_rx_sq = 4*subhalo_radhm**2 - (subhalo_posy - y_pos)**2
subhalo_dwl = ((1+z_halo)*subhalo_vmax/c )* lya


# In[8]:


ma9_index  = condition1 & (np.log10(subhalo_mass*1e10/0.6774) > 9.) & (np.log10(subhalo_mass*1e10/0.6774) < 10.)
ma10_index = condition1 & (np.log10(subhalo_mass*1e10/0.6774) > 10.) & (np.log10(subhalo_mass*1e10/0.6774) < 11.)
ma11_index = condition1 & (np.log10(subhalo_mass*1e10/0.6774) > 11.) & (np.log10(subhalo_mass*1e10/0.6774) < 12.)
ma12_index = condition1 & (np.log10(subhalo_mass*1e10/0.6774) > 12.) & (np.log10(subhalo_mass*1e10/0.6774) < 13.)
ma13_index = condition1 & (np.log10(subhalo_mass*1e10/0.6774) > 13.)

all_mass_ind=[ma9_index,ma10_index,ma11_index,ma12_index,ma13_index]


# In[9]:


subhalo_vmax[ma11_index]


# In[10]:


len(all_mass_ind[0])


# In[11]:


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


# In[12]:


#flux in wl range
flux_lya = f['flux'][:, wl_ind]
flux_map= flux_lya[y_ind_not_DLA]


# In[13]:


ind_unique = np.unique(f['ray_pos'][:,0][y_ind_not_DLA], return_index=True)
x_pos_unique = f['ray_pos'][:,0][y_ind_not_DLA][ind_unique[1]]

flux_map_unique = flux_map[ind_unique[1]]

ray_z = wavelength_all[wl_ind]

ind_sort_unique = np.argsort(x_pos_unique)
x_unique_sort = x_pos_unique[ind_sort_unique]

flux_map_unique_sort = flux_map_unique[ind_sort_unique]

# rebin to new wavelength array
wl_rebin = wave_rebin(10, ray_z)[0]
print("res:", wave_rebin(10, ray_z)[1],"km/s")

flux_rebin =  spectres.spectres(wl_rebin, ray_z, flux_map_unique_sort, fill=True, verbose=True)

#map0=np.array(np.vstack(flux_map_unique))
map0=np.array(np.vstack(flux_rebin))

#interpolate the color map to finer grid
f_map = RegularGridInterpolator((x_unique_sort, wl_rebin), map0, method='cubic')

# interpolation resolution in x~ 1 kpc, about 10* original resolution
x_grid = np.linspace(x_unique_sort[0],x_unique_sort[-1],np.int32((x_unique_sort[-1]-x_unique_sort[0])/1.0)) 
X_plot, Y_plot = np.meshgrid(x_grid[0:-1], wl_rebin[0:-1], indexing='ij')

map_plot = f_map((X_plot,Y_plot))


# In[14]:


gridspec = {'width_ratios': [1, 0.025]}
fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(15,12),gridspec_kw=gridspec)

c1 = axes2[0].pcolormesh(x_grid,wl_rebin , map_plot.T, cmap='Greys', vmin=0., vmax=1.00)

#axes2[0].scatter(subhalo_posx[cond_all],wl_halo_v0[cond_all], marker='*', s=2, c='yellow')
"""
axes2[0].errorbar(subhalo_posx[cond_all],wl_halo[cond_all],
                  xerr=subhalo_rx,yerr=subhalo_dwl, fmt='.', linewidth=1, capsize=1, c='red')

axes2[0].errorbar(np.array(halo_0_x),np.array(wl_halo_0),
                  xerr=np.array(halo_0_r),yerr=np.array(halo_0_dwl), fmt='.', linewidth=1, capsize=1, c='green')
"""

color_arr= ['green','olive','yellow','orange','red']
for i_mass in np.arange(0,len(all_mass_ind)):
    
    axes2[0].errorbar(subhalo_posx[all_mass_ind[i_mass]],wl_halo[all_mass_ind[i_mass]],
                  xerr=np.sqrt(subhalo_rx_sq[all_mass_ind[i_mass]]),yerr=subhalo_dwl[all_mass_ind[i_mass]], 
                      fmt='.', linewidth=1, capsize=1, c=color_arr[i_mass])
    

for x_i in x_los_DLA:
    axes2[0].axvline(x=x_i, ls=':', color='olive', lw=1.0, alpha=0.75)

#plot mass=12 halo
#axes2[0].scatter(np.atleast_1d(halo_0_x),np.atleast_1d(wl_halo_0_wl), marker='*', s=1, c='green')

axes2[0].set_title('flux')
# set the limits of the plot to the limits of the data
plt.colorbar(c1, cax=axes2[1])

axes2[0].set_ylabel(r'wavelength',fontsize=14)
axes2[0].set_xlabel(r'X [ckpc/h]',fontsize=14)
axes2[0].set_xlim(x_grid[0],x_grid[-1])
axes2[0].set_ylim(wl_rebin[0],wl_rebin[-1])

axes2[0].axhline(y=wl_min+5, ls='--', color='pink', lw=1.0, alpha=1.0)
axes2[0].axhline(y=wl_max-5, ls='--', color='pink', lw=1.0, alpha=1.0)

axes2[0].xaxis.set_minor_locator(AutoMinorLocator())
axes2[0].yaxis.set_minor_locator(AutoMinorLocator())
#embed()

fig2.tight_layout()
#plt.subplots_adjust(wspace=0.1, hspace=0)
plt.show()
#savename2 =('z2_all_halos_y'+str(int(y_pos))+'_mass9.05.pdf')
#plt.savefig(savename2, dpi=100)
#plt.close()


# In[15]:


gridspec = {'width_ratios': [1, 0.025]}
fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(15,12),gridspec_kw=gridspec)

c1 = axes2[0].pcolormesh(x_grid,wl_rebin , map_plot.T, cmap='Blues', vmin=0., vmax=0.25)

#axes2[0].scatter(subhalo_posx[cond_all],wl_halo_v0[cond_all], marker='*', s=2, c='yellow')
"""
axes2[0].errorbar(subhalo_posx[cond_all],wl_halo[cond_all],
                  xerr=subhalo_rx,yerr=subhalo_dwl, fmt='.', linewidth=1, capsize=1, c='red')

axes2[0].errorbar(np.array(halo_0_x),np.array(wl_halo_0),
                  xerr=np.array(halo_0_r),yerr=np.array(halo_0_dwl), fmt='.', linewidth=1, capsize=1, c='green')
"""

color_arr= ['green','olive','yellow','orange','red']
for i_mass in np.arange(0,len(all_mass_ind)):
    
    axes2[0].errorbar(subhalo_posx[all_mass_ind[i_mass]],wl_halo[all_mass_ind[i_mass]],
                 xerr=np.sqrt(subhalo_rx_sq[all_mass_ind[i_mass]]),yerr=subhalo_dwl[all_mass_ind[i_mass]], 
                      fmt='.', linewidth=1, capsize=1, c=color_arr[i_mass])
    

for x_i in x_los_DLA:
    axes2[0].axvline(x=x_i, ls=':', color='olive', lw=1.0, alpha=0.75)

#plot mass=12 halo
#axes2[0].scatter(np.atleast_1d(halo_0_x),np.atleast_1d(wl_halo_0_wl), marker='*', s=1, c='green')

axes2[0].set_title('flux')
# set the limits of the plot to the limits of the data
plt.colorbar(c1, cax=axes2[1])

axes2[0].set_ylabel(r'wavelength',fontsize=14)
axes2[0].set_xlabel(r'X [ckpc/h]',fontsize=14)
axes2[0].set_xlim(x_grid[0],x_grid[-1])
axes2[0].set_ylim(wl_rebin[0],wl_rebin[-1])

axes2[0].axhline(y=wl_min+5, ls='--', color='pink', lw=1.0, alpha=1.0)
axes2[0].axhline(y=wl_max-5, ls='--', color='pink', lw=1.0, alpha=1.0)

axes2[0].xaxis.set_minor_locator(AutoMinorLocator())
axes2[0].yaxis.set_minor_locator(AutoMinorLocator())
#embed()

fig2.tight_layout()
#plt.subplots_adjust(wspace=0.1, hspace=0)
plt.show()
#savename2 =('z2_all_halos_y'+str(int(y_pos))+'_mass9.05.pdf')
#plt.savefig(savename2, dpi=100)
#plt.close()


# In[16]:


fig3, axes3 = plt.subplots(nrows=1, ncols=2, figsize=(10,8),gridspec_kw=gridspec)

c2 = axes3[0].pcolormesh(x_grid,wl_rebin , map_plot.T, cmap='Blues', vmin=0., vmax=0.25)

color_arr= ['green','olive','yellow','orange','red']
for i_mass in np.arange(0,len(all_mass_ind)):
    
    axes2[0].errorbar(subhalo_posx[all_mass_ind[i_mass]],wl_halo[all_mass_ind[i_mass]],
                 xerr=np.sqrt(subhalo_rx_sq[all_mass_ind[i_mass]]),yerr=subhalo_dwl[all_mass_ind[i_mass]], 
                      fmt='.', linewidth=1, capsize=1, c=color_arr[i_mass])
    
x_ind_halo0_los = np.where((f['ray_pos'][y_ind_not_DLA][:,0]>=halo_0_x-halo_0_r)&(f['ray_pos'][y_ind_not_DLA][:,0]<=halo_0_x+halo_0_r))[0]
x_halo0_los = f['ray_pos'][y_ind_not_DLA][:,0][x_ind_halo0_los]

for x_i in x_los_DLA:
    axes3[0].axvline(x=x_i, ls=':', color='green', lw=1.0, alpha=0.75)

for x_i in x_halo0_los:
    axes3[0].axvline(x=x_i, ls='-', color='yellow', lw=2.0, alpha=0.5)
    
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

fig3.tight_layout()
#plt.subplots_adjust(wspace=0.1, hspace=0)
plt.show()
#savename3 =('z2_zoomin_halos_y'+str(int(y_pos))+'_mass9.05.pdf')
##\plt.savefig(savename2, dpi=100)
#plt.close()


# In[ ]:





# In[ ]:




