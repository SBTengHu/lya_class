import matplotlib.pyplot as plt
import h5py
import numpy as np
from IPython import embed

f = h5py.File('/data/forest/dsantos/DylanSims/Data/z2/KECK/Uniform/spectra_TNG50-1_z2.0_n1000d2-fullbox_KECK-HIRES-B14_HI_combined.hdf5', 'r')
wavelength_all = np.array(f['wave'])

gridspec = {'width_ratios': [1, 0.03,1, 0.03]}

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(13,5),gridspec_kw=gridspec)
ix = 781436

#make a slice of the simulation box

y_ind = np.where((f['ray_pos'][:,1]==f['ray_pos'][:,1][ix]))[0]
wl_ind = np.where((wavelength_all<1233*(1+2)) & (wavelength_all>1215.67*(1+2)))[0]
#embed()
#make the intensity plot for the slice
flux_map = f['flux'][y_ind].astype(np.float16)

im1=axes[0].scatter(f['ray_pos'][:,0],f['ray_pos'][:,1],s=10,c=f['EW_HI_1215'],cmap='binary',rasterized=True,vmax=25)
#axes[1].plot(f['ray_pos'][:,0][ix],f['ray_pos'][:,1][ix],'b*')

axes[0].scatter(f['ray_pos'][:,0][y_ind],f['ray_pos'][:,1][y_ind], marker='*', s=0.1, c='b')

cax = axes[1]
plt.colorbar(im1, cax=cax)
cax.set_ylabel(r'EW$_{Ly\alpha}$ [$\AA$]',fontsize=15)

axes[0].set_ylabel(r'Y [ckpc/h]',fontsize=14)
axes[0].set_yticks([])
axes[0].set_xlabel(r'X [ckpc/h]',fontsize=14)

axes[0].set_xlim(0,35000)
axes[0].set_ylim(0,35000)

#make the intensity plot for the slice

x_grid = np.sort(np.unique(f['ray_pos'][:,0]))
z_grid = wavelength_all[wl_ind]
map0=np.array(np.vstack(flux_map))[0:len(x_grid)-1, wl_ind[0]:wl_ind[-1]].T

#embed()

c1 = axes[2].pcolormesh(x_grid,z_grid , map0, cmap='Blues', vmin=0., vmax=1.)

axes[2].set_title('flux')
# set the limits of the plot to the limits of the data
plt.colorbar(c1, cax=axes[3])

axes[2].set_ylabel(r'wavelength',fontsize=14)
axes[2].set_xlabel(r'X [ckpc/h]',fontsize=14)
axes[2].set_xlim(x_grid[0],x_grid[-1])
axes[2].set_ylim(z_grid[0],z_grid[-1])

#embed()

fig.tight_layout()
#plt.subplots_adjust(wspace=0.1, hspace=0)
savename ='SBLA_spec_flux.pdf'

plt.savefig(savename, dpi=100)
plt.close()