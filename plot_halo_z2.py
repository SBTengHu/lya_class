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


cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)

plt.rcParams["font.family"] = "serif"
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

def restframification(wave, z):  # Function to put spectra in restframe
    wave_rest = wave / (1 + z)
    return wave_rest


def makeMatrix(xVec, yVec, cellBin):  # Function to make contours
    # Contours
    xx = np.linspace(np.min(xVec), np.max(xVec), cellBin)
    yy = np.linspace(np.min(yVec), np.max(yVec), cellBin)
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros((len(yy), len(xx)))
    for i in range(len(xx) - 1):
        for j in range(len(yy) - 1):
            nObj = (xVec > xx[i]) & (xVec < xx[i + 1]) & (yVec > yy[j]) & (yVec < yy[j + 1])
            Z[j][i] = np.log10(len(xVec[nObj]))
    return X, Y, Z


def mass_bin(mass_in, mass_f, group_mass, sbla_list):  # Function to separate halos/sblas into bins
    bins = []
    for k in range(len(sbla_list)):
        bin_col = []
        for i in range(len(mass_in)):
            bin_temp = np.where((np.log10(group_mass[mass_filter][sbla_list[k]] * 1e10 / 0.6774) >= mass_in[i]) & \
                                (np.log10(group_mass[mass_filter][sbla_list[k]] * 1e10 / 0.6774) < mass_f[i]))
            bin_col.append(bin_temp)
        bins.append(bin_col)
    return bins


def unique_globalsbla(sbla_list):  # Function to count global individual SBLAs
    sblas = []
    for k in sbla_list:
        for j in k:
            sblas.append(j)
        # print(len(list(set(u_test))))
    sblas = len(list(set(sblas)))
    return sblas


def unique_sbla(bin, sbla_list):  # Function to count individual SBLAs per bin
    unique = []
    for k in range(len(bin)):
        unique_col = []
        for n_bin in bin[k]:
            temp = []
            n_bin_sbla = sbla_list[k][n_bin]
            for all_sblas in n_bin_sbla:
                for n_sbla in all_sblas:
                    temp.append(n_sbla)
            unique_col.append(len(list(set(temp))))
        unique.append(unique_col)
    return unique


def sbla_pdf(unique, sbla_nodla, massin, massf, mass, sbla_rad):  # Fucntion to create P(SBLA)
    perc = []
    for k in range(len(unique)):
        perc_temp = np.zeros(len(unique[k]))
        for i in range(len(unique[k])):
            perc_temp[i] = unique[k][i] / len(sbla_nodla['SBLA Index'][k])
        perc.append(perc_temp)

    mass_mean = []
    mass_stdev = []
    for k in range(len(unique)):
        mean_temp = np.zeros(len(unique[k]))
        stdev_temp = np.zeros(len(unique[k]))
        for i in range(len(mass_in)):
            bin_temp = np.where((np.log10(mass[mass_filter][sbla_rad[k]] * 1e10 / 0.6774) >= massin[i]) & \
                                (np.log10(mass[mass_filter][sbla_rad[k]] * 1e10 / 0.6774) < massf[i]))
            mean_temp[i] = np.log10(np.mean(mass[mass_filter][sbla_rad[k]][bin_temp] * 1e10 / 0.6774))
            stdev_temp[i] = np.log10(np.std(mass[mass_filter][sbla_rad[k]][bin_temp] * 1e10 / 0.6774))
        mass_mean.append(mean_temp)
        mass_stdev.append(stdev_temp)
    return perc, mass_mean, mass_stdev


def halo_pdf(mass, index, bins, massin, massf):  # Function to create Halo fraction with 1 SBLA
    ratio = []
    for k in range(len(bins)):
        rat_temp = np.zeros(len(bins[k]))
        for i in range(len(bins[k])):
            total_halo = len(
                np.where((np.log10(mass * 1e10 / 0.6774) >= massin[i]) & (np.log10(mass * 1e10 / 0.6774) < massf[i]))[
                    0])
            if total_halo > 0:
                rat_temp[i] = len(index[k][bins[k][i]]) / total_halo
            else:
                rat_temp[i] = 0
        ratio.append(rat_temp)
    return ratio


def halo_pdf_cf(mass, index, bins, massin, massf, cov_fraction,
                prob):  # Function to create Halo fraction with at least prob% of covering fraction
    ratio = []
    for k in range(len(bins)):
        rat_temp = np.zeros(len(bins[k]))
        for i in range(len(bins[k])):
            total_halo = len(np.where((np.log10(mass[mass_filter] * 1e10 / 0.6774) >= massin[i]) & (
                        np.log10(mass[mass_filter] * 1e10 / 0.6774) < massf[i]))[0])
            if total_halo > 0:
                rat_temp[i] = len(cov_fraction[k][0][bins[k][i]][(cov_fraction[k][0][bins[k][i]] > prob)]) / total_halo
            else:
                rat_temp[i] = 0
        ratio.append(rat_temp)
    return ratio


def psbla_boots(sblafile, massin, massf, mass, spec_binning, rng):  # Function to boostrap P(SBLA)

    with fits.open(sblafile) as hdul:
        data = hdul[1].data

    cond = (data['Spectral Binning'] == spec_binning)
    sbla_idx = np.arange(len(data[cond]))

    with NumpyRNGContext(rng):
        sbla_idx_boots = bootstrap(sbla_idx, bootnum=100).astype(int)  # Bootstrapping all halos with SBLAs

    # P(SBLA) for original sample
    bin1 = len(np.where(data['SBLA in Mass Bin [9,9.4]'][cond] > 0)[0])
    bin2 = len(np.where(data['SBLA in Mass Bin [9.4,9.8]'][cond] > 0)[0])
    bin3 = len(np.where(data['SBLA in Mass Bin [9.8,10.2]'][cond] > 0)[0])
    bin4 = len(np.where(data['SBLA in Mass Bin [10.2,10.6]'][cond] > 0)[0])
    bin5 = len(np.where(data['SBLA in Mass Bin [10.6,11.0]'][cond] > 0)[0])
    bin6 = len(np.where(data['SBLA in Mass Bin [11,11.4]'][cond] > 0)[0])
    bin7 = len(np.where(data['SBLA in Mass Bin [11.4,11.8]'][cond] > 0)[0])
    bin8 = len(np.where(data['SBLA in Mass Bin [11.8,12.2]'][cond] > 0)[0])
    bin9 = len(np.where(data['SBLA in Mass Bin [12.2,12.6]'][cond] > 0)[0])
    bin10 = len(np.where(data['SBLA in Mass Bin [12.6,13]'][cond] > 0)[0])
    bin11 = len(np.where(data['SBLA in Mass Bin [>13]'][cond] > 0)[0])

    bins_og = np.array((bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10, bin11))
    tot_og = len(data[cond])

    psbla_og = bins_og / tot_og

    # P(SBLA) for bootsraps
    psbla_boots = []
    for i in trange(len(sbla_idx_boots)):
        cond = (data['Spectral Binning'][sbla_idx_boots[i]] == spec_binning)

        bin1 = len(np.where(data['SBLA in Mass Bin [9,9.4]'][sbla_idx_boots[i]][cond] > 0)[0])
        bin2 = len(np.where(data['SBLA in Mass Bin [9.4,9.8]'][sbla_idx_boots[i]][cond] > 0)[0])
        bin3 = len(np.where(data['SBLA in Mass Bin [9.8,10.2]'][sbla_idx_boots[i]][cond] > 0)[0])
        bin4 = len(np.where(data['SBLA in Mass Bin [10.2,10.6]'][sbla_idx_boots[i]][cond] > 0)[0])
        bin5 = len(np.where(data['SBLA in Mass Bin [10.6,11.0]'][sbla_idx_boots[i]][cond] > 0)[0])
        bin6 = len(np.where(data['SBLA in Mass Bin [11,11.4]'][sbla_idx_boots[i]][cond] > 0)[0])
        bin7 = len(np.where(data['SBLA in Mass Bin [11.4,11.8]'][sbla_idx_boots[i]][cond] > 0)[0])
        bin8 = len(np.where(data['SBLA in Mass Bin [11.8,12.2]'][sbla_idx_boots[i]][cond] > 0)[0])
        bin9 = len(np.where(data['SBLA in Mass Bin [12.2,12.6]'][sbla_idx_boots[i]][cond] > 0)[0])
        bin10 = len(np.where(data['SBLA in Mass Bin [12.6,13]'][sbla_idx_boots[i]][cond] > 0)[0])
        bin11 = len(np.where(data['SBLA in Mass Bin [>13]'][sbla_idx_boots[i]][cond] > 0)[0])

        bins_boots = np.array((bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10, bin11))
        tot_bins = len(data[sbla_idx_boots[i]][cond])

        psbla_boots.append(bins_boots / tot_bins)

    # Mass limits
    mass_mean = np.zeros(len(massin))
    for i in range(len(massin)):
        mass_mean[i] = np.log10(np.mean(mass[np.where((np.log10(mass * 1e10 / 0.6774) > massin[i]) \
                                                      & (np.log10(mass * 1e10 / 0.6774) < massf[
            i]))] * 1e10 / 0.6774))  # Mean mass per bin

    return mass_mean, psbla_og, psbla_boots


def halo_pdf_boots(mass, og_haloidx, rng, massin, massf):  # Function to calculate Halo fraction of 1 SBLA bootstraps

    halo_idx = np.arange(0, len(mass), dtype=int)  # Creating an array with all the halo indices
    with NumpyRNGContext(50):
        boots_idx = bootstrap(halo_idx, bootnum=100).astype(
            int)  # Bootstrapping these halos according to a random number generator seed
    # mbins = [9.0, 9.4, 9.8, 10.2, 10.6, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0, 13.4]
    conv_factor = 1e10 / 0.6774
    mass_conv = mass * conv_factor

    mass_mean_boots = []
    ratio_boots = []
    for idx_h in tqdm(og_haloidx):
        mass_mean_temp = []
        ratio_temp = []
        for i in trange(len(boots_idx)):
            halos_sblas = np.isin(boots_idx[i], idx_h)

            mass_mean = np.zeros(len(massin))
            ratio = np.zeros(len(massin))
            for k in range(len(massin)):
                cond_mass = (np.log10(mass_conv[boots_idx[i]]) > massin[k]) & (
                            np.log10(mass_conv[boots_idx[i]]) < massf[k])
                mass_mean[k] = np.log10(np.mean(mass_conv[boots_idx[i]][cond_mass]))

                cond_ratio_sblas = (np.log10(mass_conv[boots_idx[i][halos_sblas]]) > massin[k]) & (
                            np.log10(mass_conv[boots_idx[i][halos_sblas]]) < massf[k])
                ratio[k] = len(mass_conv[boots_idx[i][halos_sblas]][cond_ratio_sblas]) / len(
                    mass_conv[boots_idx[i]][cond_mass])

            mass_mean_temp.append(mass_mean)
            ratio_temp.append(ratio)

        mass_mean_boots.append(mass_mean_temp)
        ratio_boots.append(ratio_temp)

    return mass_mean_boots, ratio_boots


def halo_cf_pdf_boots(mass, massin, massf, cf, p, rng,
                      og_haloidx):  # Function to calculate Halo fraction with p% covering fraction bootstraps

    halo_idx = np.arange(0, len(mass), dtype=int)  # Creating an array with all the halo indices
    with NumpyRNGContext(50):
        boots_idx = bootstrap(halo_idx, bootnum=100).astype(
            int)  # Bootstrapping these halos according to a random number generator seed
    # mbins = [9.0, 9.4, 9.8, 10.2, 10.6, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0, 13.4]
    conv_factor = 1e10 / 0.6774
    mass_conv = mass * conv_factor

    mass_mean_boots = []
    ratio_boots = []
    for idx_h in tqdm(range(len(og_haloidx))):
        mass_mean_temp = []
        ratio_temp = []
        for i in trange(len(boots_idx)):
            halos_sblas = np.isin(boots_idx[i], og_haloidx[idx_h])

            mass_mean = np.zeros(len(massin))
            ratio = np.zeros(len(massin))
            for k in range(len(massin)):
                cond_mass = (np.log10(mass_conv[boots_idx[i]]) > massin[k]) & (
                            np.log10(mass_conv[boots_idx[i]]) < massf[k])
                mass_mean[k] = np.log10(np.mean(mass_conv[boots_idx[i]][cond_mass]))

                cond_ratio_sblas = (np.log10(mass_conv[boots_idx[i][halos_sblas]]) > massin[k]) & (
                            np.log10(mass_conv[boots_idx[i][halos_sblas]]) < massf[k]) & \
                                   (cf[idx_h][0][boots_idx[i][halos_sblas]] > p)
                ratio[k] = len(mass_conv[boots_idx[i][halos_sblas]][cond_ratio_sblas]) / len(
                    mass_conv[boots_idx[i]][cond_mass])

            mass_mean_temp.append(mass_mean)
            ratio_temp.append(ratio)

        mass_mean_boots.append(mass_mean_temp)
        ratio_boots.append(ratio_temp)

    return mass_mean_boots, ratio_boots


def cov_fraction_bin(covfraction, massin, massf, mass):
    mean = []
    p25 = []
    p75 = []
    for i in range(len(massin)):
        finder = np.where((np.log10(mass[mass_filter] * 1e10 / 0.6774) >= massin[i]) & (
                    np.log10(mass[mass_filter] * 1e10 / 0.6774) < massf[i]))[0]
        mean.append(np.median(covfraction[finder]))
        p25.append(np.percentile(covfraction[finder], 25))
        p75.append(np.percentile(covfraction[finder], 75))
    return mean, [p25, p75]


def wave_rebin(bins, wavelength):
    wavelength_rebin = np.zeros(int(len(wavelength) / bins))

    i = 0
    wi = 0
    while wi < len(wavelength_rebin):
        wavelength_rebin[wi] = np.sum(wavelength[i:i + bins]) / bins
        wi += 1
        i += bins

    lya = np.where((wavelength_rebin > 3640) & (wavelength_rebin < 3650))  # Lyman alpha at z=2. is 3647.01 Angstroms

    wave1 = wavelength_rebin[lya][-1]
    wave2 = wavelength_rebin[lya][-2]

    c = 2.99792e5  # km/s
    wave_lya = 1215.670  # Angstrom
    z1 = wave1 / wave_lya - 1
    z2 = wave2 / wave_lya - 1
    dz = (wave1 - wave2) / wave_lya
    deltav = c * dz / (1 + z1)

    return wavelength_rebin, deltav

# Importing Subhalo data

f = ascii.read('/data/forest/dsantos/DylanSims/Data/z2/SubhaloInfo.csv',format='csv')

subhalo_posx = np.array(f['Subhalo_PosX']) # Subhalo positions in x
subhalo_posy = np.array(f['Subhalo_PosY']) # Subhalo positions in y
subhalo_posz = np.array(f['Subhalo_PosZ']) # Subhalo positions in z
subhalo_mass = np.array(f['Subhalo_Mass']) # Subhalo mass. To convert to physical mass you have to multiply by 1e10/H0
subhalo_radhm = np.array(f['Subhalo_HalfMassRadius']) # Subhalo half mass radius. Twice this is the virial radius.
subhalo_z = np.array(f['Subhalo_GasMetal']) # Subhalo gas metallicity
subhalo_vz = np.array(f['Subhalo_PVZ']) # Subhalo peculiar velocity in z axis (km/s)
subhalo_vdisp = np.array(f['Subhalo_VDispersion']) # Subhalo velocity dispersion (km/s)
subhalo_vmax = np.array(f['Subhalo_VMax']) # Subhalo maximum velocity of the rotation curve (km/s)




f = ascii.read('/data/forest/dsantos/DylanSims/Data/z2/GroupInfo.csv',format='csv')

group_posx = np.array(f['Group_CMX']) # Group positions in x
group_posy = np.array(f['Group_CMY']) # Group positions in y
group_posz = np.array(f['Group_CMZ']) # Group positions in z
group_mass = np.array(f['Group_Mass']) # Group Mass
group_z = np.array(f['Group_Metal']) # Group Metallicity
group_vrad = np.array(f['Group_RCrit200']) # Group virial radius
group_subhaloid = np.array(f['Subhalo_ID']) # Central Subhalo ID

mass_filter = (np.log10(group_mass*1e10/0.6774) > 9.05)


f = h5py.File('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/spectra_TNG50-1_z2.0_n2000d2-rndfullbox_SDSS-BOSS_HI_combined.hdf5', 'r')
wavelength = np.array(f['wave'])

# Rebinning wavelengths and each spectral binning velocity
wave_rebins = [wave_rebin(2, wavelength)[0], wave_rebin(3, wavelength)[0], wave_rebin(4, wavelength)[0], wave_rebin(6, wavelength)[0]]
vel = [wave_rebin(2, wavelength)[1], wave_rebin(3, wavelength)[1], wave_rebin(4, wavelength)[1], wave_rebin(6, wavelength)[1]]
vel_str = [str(round(elements,0)) for elements in vel]

# List of all SBLAs per flux and spectral binning
with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/SBLA_025_AllBins_Index.fits') as hdul:
    sbla_025 = hdul[1].data

with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/SBLA_015_AllBins_Index.fits') as hdul:
    sbla_015 = hdul[1].data

with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/SBLA_005_AllBins_Index.fits') as hdul:
    sbla_005 = hdul[1].data

sbla_lists = [sbla_025, sbla_015, sbla_005]

# List of SBLAs in Halos

sbla025_vmax_vrad = {}
sbla015_vmax_vrad = {}
sbla005_vmax_vrad = {}

for i in trange(len(vel_str)):
    with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/Index/SBLA025_' + vel_str[
        i] + '_vmax_vrad.fits') as hdul:
        data = hdul[1].data
        t1 = data['Subhalo Index']
        t2 = data['SBLA Indices']

    if i == 0:
        sbla025_vmax_vrad['Halo Index'] = [t1]
        sbla025_vmax_vrad['SBLAs Indices'] = [t2]
    else:
        sbla025_vmax_vrad['Halo Index'].append(t1)
        sbla025_vmax_vrad['SBLAs Indices'].append(t2)

    with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/Index/SBLA015_' + vel_str[
        i] + '_vmax_vrad.fits') as hdul:
        data = hdul[1].data
        t1 = data['Subhalo Index']
        t2 = data['SBLA Indices']

    if i == 0:
        sbla015_vmax_vrad['Halo Index'] = [t1]
        sbla015_vmax_vrad['SBLAs Indices'] = [t2]
    else:
        sbla015_vmax_vrad['Halo Index'].append(t1)
        sbla015_vmax_vrad['SBLAs Indices'].append(t2)

    with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/Index/SBLA005_' + vel_str[
        i] + '_vmax_vrad.fits') as hdul:
        data = hdul[1].data
        t1 = data['Subhalo Index']
        t2 = data['SBLA Indices']

    if i == 0:
        sbla005_vmax_vrad['Halo Index'] = [t1]
        sbla005_vmax_vrad['SBLAs Indices'] = [t2]
    else:
        sbla005_vmax_vrad['Halo Index'].append(t1)
        sbla005_vmax_vrad['SBLAs Indices'].append(t2)

sbla025_vdisp_vrad = {}
sbla015_vdisp_vrad = {}
sbla005_vdisp_vrad = {}

for i in trange(len(vel_str)):
    with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/Index/SBLA025_' + vel_str[
        i] + '_vdisp_vrad.fits') as hdul:
        data = hdul[1].data
        t1 = data['Subhalo Index']
        t2 = data['SBLA Indices']

    if i == 0:
        sbla025_vdisp_vrad['Halo Index'] = [t1]
        sbla025_vdisp_vrad['SBLAs Indices'] = [t2]
    else:
        sbla025_vdisp_vrad['Halo Index'].append(t1)
        sbla025_vdisp_vrad['SBLAs Indices'].append(t2)

    with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/Index/SBLA015_' + vel_str[
        i] + '_vdisp_vrad.fits') as hdul:
        data = hdul[1].data
        t1 = data['Subhalo Index']
        t2 = data['SBLA Indices']

    if i == 0:
        sbla015_vdisp_vrad['Halo Index'] = [t1]
        sbla015_vdisp_vrad['SBLAs Indices'] = [t2]
    else:
        sbla015_vdisp_vrad['Halo Index'].append(t1)
        sbla015_vdisp_vrad['SBLAs Indices'].append(t2)

    with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/Index/SBLA005_' + vel_str[
        i] + '_vdisp_vrad.fits') as hdul:
        data = hdul[1].data
        t1 = data['Subhalo Index']
        t2 = data['SBLA Indices']

    if i == 0:
        sbla005_vdisp_vrad['Halo Index'] = [t1]
        sbla005_vdisp_vrad['SBLAs Indices'] = [t2]
    else:
        sbla005_vdisp_vrad['Halo Index'].append(t1)
        sbla005_vdisp_vrad['SBLAs Indices'].append(t2)

sbla025_vmax_vrad = pd.DataFrame.from_dict(sbla025_vmax_vrad)
sbla015_vmax_vrad = pd.DataFrame.from_dict(sbla015_vmax_vrad)
sbla005_vmax_vrad = pd.DataFrame.from_dict(sbla005_vmax_vrad)

sbla025_vdisp_vrad = pd.DataFrame.from_dict(sbla025_vdisp_vrad)
sbla015_vdisp_vrad = pd.DataFrame.from_dict(sbla015_vdisp_vrad)
sbla005_vdisp_vrad = pd.DataFrame.from_dict(sbla005_vdisp_vrad)

# Lists of SBLA properties
prop_files = ['/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/SBLAProps_025_vmax.fits',
              '/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/SBLAProps_015_vmax.fits',
              '/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/SBLAProps_005_vmax.fits',
             '/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/SBLAProps_025_vdisp.fits',
              '/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/SBLAProps_015_vdisp.fits',
              '/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/SBLAProps_005_vdisp.fits']

# Importing covering fractions

with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/CovFraction_vmax_025.fits') as hdul:
        cov_fraction_vmax_025 = hdul[1].data

with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/CovFraction_vmax_015.fits') as hdul:
        cov_fraction_vmax_015 = hdul[1].data

with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/CovFraction_vmax_005.fits') as hdul:
        cov_fraction_vmax_005 = hdul[1].data

with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/CovFraction_vdisp_025.fits') as hdul:
        cov_fraction_vdisp_025 = hdul[1].data

with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/CovFraction_vdisp_015.fits') as hdul:
        cov_fraction_vdisp_015 = hdul[1].data

with fits.open('/data/forest/dsantos/DylanSims/Data/z2/SDSS/Random/DLA_SBLA/CovFraction_vdisp_005.fits') as hdul:
        cov_fraction_vdisp_005 = hdul[1].data

bins = []
mass_in = []
mass_f = []
temp_mass = 9.

while temp_mass < max(np.log10(group_mass[mass_filter] * 1e10 / 0.6774)):
    mass_in.append(temp_mass)
    bin_temp_vmax = np.where((np.log10(group_mass[mass_filter] * 1e10 / 0.6774) >= temp_mass) & \
                             (np.log10(group_mass[mass_filter] * 1e10 / 0.6774) < temp_mass + 0.4))
    temp_mass += 0.4

    bins.append(bin_temp_vmax[0])
    mass_f.append(temp_mass)

mass_in = np.array(mass_in)
mass_f = np.array(mass_f)

fig = plt.figure(figsize=(7, 5))

halos_per_bin = []

for i in bins:
    halos_per_bin.append(len(i))

plt.plot((mass_in + mass_f) / 2, np.log10(halos_per_bin), '-k.')
for i in range(11):
    plt.text((mass_in[i] + mass_f[i]) / 2, np.log10(halos_per_bin[i]), str(halos_per_bin[i]))

plt.xlabel('Halo Mass', fontsize=15)
plt.ylabel(r'$\log_{10}$(Halo Counts)', fontsize=15)

fig.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0)
plt.savefig('z2_plots/AllHaloBins.pdf', bbox_inches='tight', dpi=300)
plt.savefig('z2_plots/AllHaloBins.png', bbox_inches='tight', dpi=300)
plt.show()

bins025_vmax_vrad = mass_bin(mass_in, mass_f, group_mass, sbla025_vmax_vrad['Halo Index'])
bins015_vmax_vrad = mass_bin(mass_in, mass_f, group_mass,  sbla015_vmax_vrad['Halo Index'])
bins005_vmax_vrad = mass_bin(mass_in, mass_f, group_mass, sbla005_vmax_vrad['Halo Index'])

bins025_vdisp_vrad = mass_bin(mass_in, mass_f, group_mass, sbla025_vdisp_vrad['Halo Index'])
bins015_vdisp_vrad = mass_bin(mass_in, mass_f, group_mass,  sbla015_vdisp_vrad['Halo Index'])
bins005_vdisp_vrad = mass_bin(mass_in, mass_f, group_mass, sbla005_vdisp_vrad['Halo Index'])

all_sblas = []

for k in range(len(sbla_lists)):
    for j in range(len(sbla_lists[k])):
        all_sblas.append(len(sbla_lists[k]['SBLA Index'][j]))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9,4),sharey='all',sharex='all')

vels = [r'$v_{max}$', r'$v_{disp}$']

lstyle = ['-','--','-.',':']

for i in range(1):
    axes[0].plot(vels, [unique_globalsbla(sbla025_vmax_vrad['SBLAs Indices'][i])/all_sblas[i], unique_globalsbla(sbla025_vdisp_vrad['SBLAs Indices'][i])/all_sblas[i]],\
                        color='k', linestyle=lstyle[i], label=vel_str[i]+'km/s')

#######################################################################################################################

for i in range(1):
    axes[1].plot(vels, [unique_globalsbla(sbla015_vmax_vrad['SBLAs Indices'][i])/all_sblas[i+4], unique_globalsbla(sbla015_vdisp_vrad['SBLAs Indices'][i])/all_sblas[i+4]],\
                        color='b', linestyle=lstyle[i], label=vel_str[i]+'km/s')

# # #######################################################################################################################

for i in range(1):
    axes[2].plot(vels, [unique_globalsbla(sbla005_vmax_vrad['SBLAs Indices'][i])/all_sblas[i+4*2], unique_globalsbla(sbla005_vdisp_vrad['SBLAs Indices'][i])/all_sblas[i+4*2]],\
                        color='g', linestyle=lstyle[i], label=vel_str[i]+'km/s')

axes[0].set_ylabel(r'P(SBLA)$_{global}$', fontsize=15)

# axes[0].legend()

for i in range(0,3):
    axes[i].set_xticklabels(vels, rotation=90, ha='right')

axes[0].set_title(r'$F_{Ly\alpha} < 0.25$')
axes[1].set_title(r'$F_{Ly\alpha} < 0.15$')
axes[2].set_title(r'$F_{Ly\alpha} < 0.05$')

fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('z2_plots/PSBLA_Global_Random.pdf', bbox_inches='tight', dpi=300)
plt.savefig('z2_plots/PSBLA_Global_Random.png', bbox_inches='tight', dpi=300)
plt.show()

ig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,6),sharey='all',sharex='all')

lstyle = ['-','--','-.',':']

unique_vmax_025 = unique_sbla(bins025_vmax_vrad,sbla025_vmax_vrad['SBLAs Indices'])
fsbla = sbla_pdf(unique_vmax_025,sbla_025, mass_in, mass_f, group_mass, sbla025_vmax_vrad['Halo Index'])

# for i in range(len(fsbla[1])):
axes[0,0].plot(fsbla[1][0][:-1], fsbla[0][0][:-1], color='k', linestyle=lstyle[0])#, label=vel_str[0]+' km/s')
# axes[0,0].errorbar(fsbla[1][0][:-1], fsbla[0][0][:-1], yerr=boots_deviation[0][:-1], capsize=0.5, color='k', linestyle='')

###

unique_vmax_015 = unique_sbla(bins015_vmax_vrad,sbla015_vmax_vrad['SBLAs Indices'])
fsbla = sbla_pdf(unique_vmax_015,sbla_015,mass_in, mass_f, group_mass, sbla015_vmax_vrad['Halo Index'])

# for i in range(len(fsbla[1])):
axes[0,1].plot(fsbla[1][0][:-1], fsbla[0][0][:-1], color='b', linestyle=lstyle[0], label=vel_str[0]+'km/s')
# axes[0,1].errorbar(fsbla[1][0][:-1], fsbla[0][0][:-1], yerr=boots_deviation[0+4][:-1], capsize=0.5, color='b', linestyle='')

###

unique_vmax_005 = unique_sbla(bins005_vmax_vrad,sbla005_vmax_vrad['SBLAs Indices'])
fsbla = sbla_pdf(unique_vmax_005,sbla_005,mass_in, mass_f, group_mass, sbla005_vmax_vrad['Halo Index'])

# for i in range(len(fsbla[1])):
axes[0,2].plot(fsbla[1][0][:-1], fsbla[0][0][:-1], color='g', linestyle=lstyle[0], label=vel_str[0]+' km/s')
# axes[0,2].errorbar(fsbla[1][0][:-1], fsbla[0][0][:-1], yerr=boots_deviation[0+4*2][:-1], capsize=0.5, color='g', linestyle='')

# ##########################################################################################################

unique_vdisp_025 = unique_sbla(bins025_vdisp_vrad,sbla025_vdisp_vrad['SBLAs Indices'])
fsbla = sbla_pdf(unique_vdisp_025,sbla_025,mass_in, mass_f, group_mass, sbla025_vdisp_vrad['Halo Index'])

# for i in range(len(fsbla[1])):
axes[1,0].plot(fsbla[1][0][:-1], fsbla[0][0][:-1], color='k', linestyle=lstyle[0], label=vel_str[0])
# axes[1,0].errorbar(fsbla[1][0][:-1], fsbla[0][0][:-1], yerr=boots_deviation[0+4*3][:-1], capsize=0.5, color='k', linestyle='')

###

unique_vdisp_015 = unique_sbla(bins015_vdisp_vrad,sbla015_vdisp_vrad['SBLAs Indices'])
fsbla = sbla_pdf(unique_vdisp_015,sbla_015,mass_in, mass_f, group_mass, sbla015_vdisp_vrad['Halo Index'])

# for i in range(len(fsbla[1])):
axes[1,1].plot(fsbla[1][0][:-1], fsbla[0][0][:-1], color='b', linestyle=lstyle[0], label=vel_str[0])
# axes[1,1].errorbar(fsbla[1][0][:-1], fsbla[0][0][:-1], yerr=boots_deviation[0+4*4][:-1], capsize=0.5, color='b', linestyle='')

###

unique_vdisp_005 = unique_sbla(bins005_vdisp_vrad,sbla005_vdisp_vrad['SBLAs Indices'])
fsbla = sbla_pdf(unique_vdisp_005,sbla_005,mass_in, mass_f, group_mass, sbla005_vdisp_vrad['Halo Index'])

# for i in range(len(fsbla[1])):
axes[1,2].plot(fsbla[1][0][:-1], fsbla[0][0][:-1], color='g', linestyle=lstyle[0], label=vel_str[0]+' km/s')
# axes[1,2].errorbar(fsbla[1][0][:-1], fsbla[0][0][:-1], yerr=boots_deviation[0+4*5][:-1], capsize=0.5, color='g', linestyle='')

label = [r'$v_{max}$', r'$v_{disp}$']

for i in range(2):
    ax1 = axes[i,2].twinx()
    ax1.set_yticks([])
    ax1.set_ylabel(label[i], fontsize=15, rotation=270,labelpad=15)

for i in range(0,3):
    for j in range(0,2):
        axes[j,i].axvline(np.log10(1e12/0.6781),color='firebrick',alpha=0.5, linestyle='--', label='SBLA Bias')
        axes[j,i].axvline(np.log10(4e11/0.6781),color='firebrick',alpha=0.5, label='138 km/s $v_{circ}$')

axes[0,0].set_ylabel('P(SBLA)', fontsize=15)
axes[1,0].set_ylabel('P(SBLA)', fontsize=15)
axes[1,0].set_xlabel(r'Halo Mass [log($M_{\odot}$)]', fontsize=15)
axes[1,1].set_xlabel(r'Halo Mass [log($M_{\odot}$)]', fontsize=15)
axes[1,2].set_xlabel(r'Halo Mass [log($M_{\odot}$)]', fontsize=15)

axes[0,0].legend(ncol=1)

axes[0,0].set_title(r'$F_{Ly\alpha} < 0.25$')
axes[0,1].set_title(r'$F_{Ly\alpha} < 0.15$')
axes[0,2].set_title(r'$F_{Ly\alpha} < 0.05$')

axes[0,0].set_xticks(np.arange(9, 13.1, 0.5))
axes[0,0].set_xlim(9.1,12.9)

fig.tight_layout()
plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.savefig('z2_plots/PSBLA_Bins_Rand.pdf', bbox_inches='tight', dpi=300)
plt.savefig('z2_plots/PSBLA_Bins_Rand.png', bbox_inches='tight', dpi=300)
plt.show()