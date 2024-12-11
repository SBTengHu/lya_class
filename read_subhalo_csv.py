import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits, ascii

f = ascii.read('/data/forest/dsantos/DylanSims/Data/z2/SubhaloInfo.csv',format='csv')
subhalo_posx = np.array(f['Subhalo_PosX']) # Subhalo positions in x
subhalo_posy = np.array(f['Subhalo_PosY']) # Subhalo positions in y
subhalo_posz = np.array(f['Subhalo_PosZ']) # Subhalo positions in z
subhalo_mass = np.array(f['Subhalo_Mass']) # Subhalo mass. To convert to physical mass you have to multiply by 1e10/H0
subhalo_radhm = np.array(f['Subhalo_HalfMassRadius']) # Subhalo half mass radius. Twice this is the virial radius.
subhalo_z = np.array(f['Subhalo_GasMetal']) # Subhalo gas metallicity
subhalo_vz = np.array(f['Subhalo_PVZ']) # Subhalo peculiar velocity in z axis (km/s)
subhalo_vdisp = np.array(f['Subhalo_VDispersion']) # Subhalo velocity dispersion (km/s)
ubhalo_vmax = np.array(f['Subhalo_VMax']) # Subhalo maximum velocity of the rotation curve (km/s)

dataout = np.column_stack((subhalo_posx, subhalo_posy, subhalo_posz, subhalo_mass, subhalo_radhm,
                            subhalo_vz, subhalo_vdisp))
header = "subhalo_posx, subhalo_posy, subhalo_posz, subhalo_mass, subhalo_radhm,subhalo_vz, subhalo_vdisp"
np.savetxt('subhalo', dataout, header=header)