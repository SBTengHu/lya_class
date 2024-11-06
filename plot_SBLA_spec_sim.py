import matplotlib.pyplot as plt
import h5py
import numpy as np
from IPython import embed

f = h5py.File('/data/forest/dsantos/DylanSims/Data/z2/KECK/Uniform/spectra_TNG50-1_z2.0_n1000d2-fullbox_KECK-HIRES-B14_HI_combined.hdf5', 'r')
wavelength = np.array(f['wave'])

gridspec = {'width_ratios': [1, 1, 0.05]}
ix = 781436

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,5),gridspec_kw=gridspec)

axes[0].plot(wavelength,f['flux'][ix],'b-')

axes[0].set_xlim(1215.67*(1+2),1233*(1+2))
axes[0].set_ylim(0.,1.05)

embed()
im1=axes[1].scatter(f['ray_pos'][:,0],f['ray_pos'][:,1],s=10,c=f['EW_HI_1215'],cmap='binary',rasterized=True,vmax=25)
axes[1].plot(f['ray_pos'][:,0][ix],f['ray_pos'][:,1][ix],'b*')

cax = axes[2]
plt.colorbar(im1, cax=cax)
cax.set_ylabel(r'EW$_{Ly\alpha}$ [$\AA$]',fontsize=15)

axes[1].set_ylabel(r'Y [ckpc/h]',fontsize=14)
axes[1].set_yticks([])
axes[1].set_xlabel(r'X [ckpc/h]',fontsize=14)

axes[1].set_xlim(0,35000)
axes[1].set_ylim(0,35000)

axes[0].set_xlabel(r'Observed Wavelength [$\AA$]', fontsize=15)
axes[0].set_ylabel(r'Normalised Flux', fontsize=15)

fig.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0)
savename ='SBLA_spec.pdf'
plt.savefig(savename, dpi=1300)
plt.close()