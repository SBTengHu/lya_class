import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits, ascii
import h5py
from scipy.interpolate import interp2d

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
hubblez = cosmo.H(z_0)

#for plot
gridspec = {'width_ratios': [1, 0.025]}

f = h5py.File('/data/forest/dsantos/DylanSims/Data/z2/KECK/Random/spectra_TNG50-1_z2.0_n2000d2-rndfullbox_KECK-HIRES-B14_HI_combined.hdf5', 'r')
#f = h5py.File('/data/forest/dsantos/DylanSims/Data/z2/KECK/Uniform/spectra_TNG50-1_z2.0_n1000d2-fullbox_KECK-HIRES-B14_HI_combined.hdf5', 'r')
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
ubhalo_vmax = np.array(f1['Subhalo_VMax']) # Subhalo maximum velocity of the rotation curve (km/s)

#f = ascii.read('/data/forest/dsantos/DylanSims/Data/z2/GroupInfo.csv',format='csv')
#group_posx = np.array(f['Group_CMX']) # Group positions in x
#group_posy = np.array(f['Group_CMY']) # Group positions in y
#group_posz = np.array(f['Group_CMZ']) # Group positions in z
#group_mass = np.array(f['Group_Mass']) # Group Mass
#group_z = np.array(f['Group_Metal']) # Group Metallicity
#group_vrad = np.array(f['Group_RCrit200']) # Group virial radius
#group_subhaloid = np.array(f['Subhalo_ID']) # Central Subhalo ID

mass_filter_0 = (np.log10(subhalo_mass*1e10/0.6774) > 12.45) & (np.log10(subhalo_mass*1e10/0.6774) < 12.55)
y_pos = subhalo_posy[mass_filter_0][0]

x_grid = np.linspace(0,35000,3*ncell)

#condition if for all subhalos in the plane of the slice
condition1 = (subhalo_posy - y_pos)**2  <= (2*subhalo_radhm)**2
condition2 = (np.log10(subhalo_mass*1e10/0.6774) > 9.05)

cond_all = condition1 & condition2

#the radius projected in x-axis
subhalo_rx= np.zeros(len(subhalo_posx[cond_all]))
subhalo_rx = np.sqrt(4*subhalo_radhm[cond_all]**2 - (subhalo_posy[cond_all] - y_pos)**2)
subhalo_dwl = subhalo_vdisp[cond_all]/c * lya

#the width of slice in kpc/h
dy=35

y_ind = np.where((f['ray_pos'][:,1]>=y_pos-dy/2)&(f['ray_pos'][:,1]<=y_pos+dy/2))[0]
wl_ind = np.where((wavelength_all<1233*(1+2)) & (wavelength_all>1215.67*(1+2)))[0]

flux_map_allwl = f['flux'][y_ind][wl_ind]


"""
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,5),gridspec_kw=gridspec)
im1=axes[0].scatter(f['ray_pos'][:,0],f['ray_pos'][:,1],s=10,c=f['EW_HI_1215'],cmap='binary',rasterized=True,vmax=25)
#axes[1].plot(f['ray_pos'][:,0][ix],f['ray_pos'][:,1][ix],'b*')

#axes[0].scatter(f['ray_pos'][:,0][y_ind],f['ray_pos'][:,1][y_ind], marker='*', s=0.1, c='b')
cax = axes[1]
plt.colorbar(im1, cax=cax)

cax.set_ylabel(r'EW$_{Ly\alpha}$ [$\AA$]',fontsize=15)

axes[0].set_ylabel(r'Y [ckpc/h]',fontsize=14)
axes[0].set_yticks([])
axes[0].set_xlabel(r'X [ckpc/h]',fontsize=14)
axes[0].set_xlim(0,35000)
axes[0].set_ylim(0,35000)

fig.tight_layout()
#plt.subplots_adjust(wspace=0.1, hspace=0)
savename ='sim_cut.pdf'
plt.savefig(savename, dpi=100)
plt.close()
"""

dz = ((Lbox / h / ncell) * hubblez / c).value # dz in redshift per cell
dWL = (dz * lya)  # wl in AA per cell
dl = Lbox / ncell  # dist per cell for raw skewer in Mpc/h


mass_filter = (np.log10(group_mass*1e10/0.6774) > 9.05)

#condition if for all subhalos in the plane of the slice
condition1 = (group_posy - y_pos)**2  <= group_vrad**2
condition2 = mass_filter

cond_all = condition1 #& condition2


#convert from z position to wavelength
z_halo = group_posz/(dl*1000*h)  * dz+ (2.0)
wl_halo = (1+z_halo )* lya
#wl_halo = (1+z_halo + subhalo_vz/c )* lya
#wl_halo_min = wl_halo - subhalo_vdisp/c * lya
#wl_halo_max = wl_halo + subhalo_vdisp/c * lya

#make the intensity plot for the slice

x_grid = np.sort(np.unique(f['ray_pos'][:,0]))
z_grid = wavelength_all[wl_ind]
map0=np.array(np.vstack(flux_map))[0:len(x_grid)-1, wl_ind[0]:wl_ind[-1]].T

#embed()

fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(13,5),gridspec_kw=gridspec)

c1 = axes2[0].pcolormesh(x_grid,z_grid , map0, cmap='Blues', vmin=0., vmax=1.)

axes2[0].scatter(group_posx[cond_all],wl_halo[cond_all], marker='*', s=2, c='red')

axes2[0].set_title('flux')
# set the limits of the plot to the limits of the data
plt.colorbar(c1, cax=axes2[1])

axes2[0].set_ylabel(r'wavelength',fontsize=14)
axes2[0].set_xlabel(r'X [ckpc/h]',fontsize=14)
axes2[0].set_xlim(x_grid[0],x_grid[-1])
axes2[0].set_ylim(z_grid[0],z_grid[-1])

#embed()

fig2.tight_layout()
#plt.subplots_adjust(wspace=0.1, hspace=0)
plt.show()


savename2 =('SBLA_flux_halos.pdf')
plt.savefig(savename2, dpi=100)
plt.close()