import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits, ascii
import h5py
from scipy.interpolate import interp2d, RegularGridInterpolator
from matplotlib.ticker import AutoMinorLocator,MaxNLocator
import astropy.table as tab
from sys import getsizeof


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

dz = ((Lbox / h / ncell) * hubblez / c).value # dz in redshift per cell
dWL = (dz * lya)  # wl in AA per cell
dl = Lbox / ncell  # dist per cell for raw skewer in Mpc/h

from scipy.spatial import cKDTree

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
subhalo_vmax = np.array(f1['Subhalo_VMax']) # Subhalo maximum velocity of the rotation curve (km/s)

#f = ascii.read('/data/forest/dsantos/DylanSims/Data/z2/GroupInfo.csv',format='csv')
#group_posx = np.array(f['Group_CMX']) # Group positions in x
#group_posy = np.array(f['Group_CMY']) # Group positions in y
#group_posz = np.array(f['Group_CMZ']) # Group positions in z
#group_mass = np.array(f['Group_Mass']) # Group Mass
#group_z = np.array(f['Group_Metal']) # Group Metallicity
#group_vrad = np.array(f['Group_RCrit200']) # Group virial radius
#group_subhaloid = np.array(f['Subhalo_ID']) # Central Subhalo ID


DLA_list ='/data/forest/dsantos/DylanSims/Data/z2/KECK/Random/Old_SpecBins/DLA_SBLA_HighRes/DLA_AllBins_Index.fits'
f_DLA=tab.Table.read(DLA_list, hdu=1)

#mark DLA index
all_indices = np.arange(len(f['flux']))
not_DLA = np.isin(all_indices, f_DLA[0][0], invert=True)
ind_not_DLA = all_indices[not_DLA]

#convert from z position to wavelength
z_halo = subhalo_posz/(dl*1000)  * dz+ (z_0)
#wl_halo = (1+z_halo )* lya
wl_halo = (1+z_halo + subhalo_vz/c )* lya

# extendthe bourandary a little bit
wl_min = (1+z_0)*lya-5
wl_max = (1+z_0 + 1000*dz)*lya+5

wl_ind = np.where((wavelength_all<wl_max) & (wavelength_all>wl_min))[0]

ray_z = wavelength_all[wl_ind]

condition2 = (np.log10(subhalo_mass*1e10/0.6774) > 9.05)

# Assuming f, subhalo_posx, subhalo_posy, subhalo_radhm, subhalo_vz, subhalo_vmax, z_halo, lya, c, and ray_z are already defined
LOS_xy = np.array((f['ray_pos'][ind_not_DLA][:, 0], f['ray_pos'][ind_not_DLA][:, 1])).T
subhalo_positions = np.array((subhalo_posx[condition2], subhalo_posy[condition2])).T
subhalo_radii = 2 * subhalo_radhm[condition2]
z_halo_cond = z_halo[condition2]
subhalo_vz_cond = subhalo_vz[condition2]
subhalo_vmax_cond = subhalo_vmax[condition2]

# Create a KDTree for LOS_xy
tree = cKDTree(LOS_xy)

# Initialize lists to store results
los_halo_ind = []
wl_halo_ind_all = []

# Iterate over each halo
for i, (pos, radius) in enumerate(zip(subhalo_positions, subhalo_radii)):
    # Find all LOS within the radius in the x-y plane
    los_indices0 = tree.query_ball_point(pos, radius)

    # Store the results
    los_halo_ind.append(los_indices0)

    wl_los_halo_indices = []
    for j in los_indices0:
        # Calculate the wavelength range for the halo in the z-direction
        halo_dwl_min = (1 + z_halo_cond[i] + subhalo_vz_cond[i] / c - subhalo_vmax_cond[i] / c) * lya
        halo_dwl_max = (1 + z_halo_cond[i] + subhalo_vz_cond[i] / c + subhalo_vmax_cond[i] / c) * lya

        # Find the wavelength indices that are covered by the halo
        wl_indices0 = np.where((ray_z >= halo_dwl_min) & (ray_z <= halo_dwl_max))[0]

        # Store the results
        wl_los_halo_indices.append(wl_indices0)

    wl_halo_ind_all.append(wl_los_halo_indices)

# los_indices now contains the indices of LOS in the x-y plane for each halo
# wavelength_indices contains the wavelength indices covered by each halo in the z-direction

# Create the LOS_MASK array
LOS_MASK = np.zeros((len(f['ray_pos'][ind_not_DLA]), len(ray_z)), dtype=float)

# Set the specified elements to 1
for los, wl_list in zip(los_halo_ind, wl_halo_ind_all):
    for l, wl in zip(los, wl_list):
        LOS_MASK[l, wl] = 1

# generate a dataset for ML
flux_lya = f['flux'][[ind_not_DLA]][:, wl_ind]

# Save flux_subset and LOS_MASK to an HDF5 file
with h5py.File('SBLA_flux_z'+str(z_0)+'_spec'+str(len(flux_lya))+'.hdf5', 'w') as hf:
    hf.create_dataset('flux_lya', data=flux_lya)
    hf.create_dataset('LOS_MASK', data=LOS_MASK)
    #restore the wavelength
    hf.create_dataset('wavelength_range', data=ray_z)
print("Data saved to output.hdf5")

# readin the data
#with h5py.File('output.hdf5', 'r') as hf:
#    flux_lya = hf['flux_lya'][:]
#    LOS_MASK = hf['LOS_MASK'][:]
