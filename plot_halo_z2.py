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


test test




