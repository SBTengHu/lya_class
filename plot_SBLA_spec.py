import matplotlib.pyplot as plt
import h5py
import numpy as np
from tqdm.notebook import trange, tqdm
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import z_at_value
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
from collections import Counter
import spectres
import pandas as pd
from IPython import embed

f = h5py.File('/data/forest/dsantos/DylanSims/Data/z2/KECK/Uniform/spectra_TNG50-1_z2.0_n1000d2-fullbox_KECK-HIRES-B14_HI_combined.hdf5', 'r')
wavelength = np.array(f['wave'])

gridspec = {'width_ratios': [1, 0.05]}
ix = 781436

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,5),gridspec_kw=gridspec)

axes[0].plot(wavelength,f['flux'][ix],'b-')

axes[0].set_xlim(1215.67*(1+2),1233*(1+2))
axes[0].set_ylim(0.,1.05)

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
plt.show()

halo_0_los_arr = f['flux'][x_ind_halo0_los]

for x in x_ind_halo0_los[0:3]:
    flux = f['flux'][x]
    wave = wavelength_all

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.plot(wave, flux,
            color='0.5', drawstyle='steps', zorder=1, lw=0.5)

    ax.axvspan(wl_halo_0_min, wl_halo_0_max, alpha=0.5, color='red')

    ax.set_ylabel('Transmission')
    ax.set_xlabel('$\\lambda / \\AA$')

    ax.set_ylim(-0.1, 1.25)
    ax.set_xlim(ray_z[0], ray_z[-1])

    plt.savefig('halo_los_spec' + str(x) + '_plot.pdf')
    plt.close()