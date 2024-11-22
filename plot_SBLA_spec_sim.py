import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits, ascii
import h5py

f = h5py.File('/data/forest/dsantos/DylanSims/Data/z2/KECK/Uniform/spectra_TNG50-1_z2.0_n1000d2-fullbox_KECK-HIRES-B14_HI_combined.hdf5', 'r')
wavelength_all = np.array(f['wave'])

gridspec = {'width_ratios': [1, 0.05]}

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,5),gridspec_kw=gridspec)

ix = 781436

#make a slice of the simulation box

y_ind = np.where((f['ray_pos'][:,1]==f['ray_pos'][:,1][ix]))[0]
wl_ind = np.where((wavelength_all<1233*(1+2)) & (wavelength_all>1215.67*(1+2)))[0]

y_pos = f['ray_pos'][:,1][ix]
#embed()
#make the intensity plot for the slice
flux_map = f['flux'][y_ind].astype(np.float16)

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


#cosmological parameters (not correct)
c = 3e5  # km/s
lya = 1215.67  # AA
h = 0.700
OmegaM = 0.30
OmegaB = 0.05
z = 2.0
Lbox = 50  # Mpc/h
ncell = 1000#len(np.sort(np.unique(f['ray_pos'][:,0])))
cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=OmegaM, Ob0=OmegaB)
hubblez = cosmo.H(z)

dz = ((Lbox / h / ncell) * hubblez / c).value # dz in redshift per cell
dWL = (dz * lya)  # wl in AA per cell
dl = Lbox / ncell  # dist per cell for raw skewer in Mpc/h

# "subhalo_posx, subhalo_posy, subhalo_posz, subhalo_mass, subhalo_radhm,subhalo_vz, subhalo_vdisp"
dataf1='/net/CLUSTER/VAULT/users/thu/subhalo'

subhalo_posx, subhalo_posy, subhalo_posz, subhalo_mass, subhalo_radhm,subhalo_vz, subhalo_vdisp\
    = np.loadtxt(dataf1, unpack=1, dtype='float64')
# "subhalo_posx, subhalo_posy, subhalo_posz, subhalo_mass, subhalo_radhm,subhalo_vz, subhalo_vdisp"

#condition if for all subhalos in the plane of the slice
condition1 = (subhalo_posy - y_pos)**2  <= subhalo_radhm**2
condition2 = subhalo_mass >= 1e2

cond_all = condition1 #& condition2


#convert from z position to wavelength
z_halo = subhalo_posz/(dl*1000*h)  * dz+ (2.0)
#wl_halo = (1+z_halo )* lya
wl_halo = (1+z_halo + subhalo_vz/c )* lya
wl_halo_min = wl_halo - subhalo_vdisp/c * lya
wl_halo_max = wl_halo + subhalo_vdisp/c * lya

#make the intensity plot for the slice

x_grid = np.sort(np.unique(f['ray_pos'][:,0]))
z_grid = wavelength_all[wl_ind]
map0=np.array(np.vstack(flux_map))[0:len(x_grid)-1, wl_ind[0]:wl_ind[-1]].T

#embed()

fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(13,5),gridspec_kw=gridspec)

c1 = axes2[0].pcolormesh(x_grid,z_grid , map0, cmap='Blues', vmin=0., vmax=1.)

axes2[0].scatter(subhalo_posx[cond_all],wl_halo[cond_all], marker='*', s=2, c='red')

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
savename2 =('SBLA_flux_halos.pdf')

plt.savefig(savename2, dpi=100)
plt.close()