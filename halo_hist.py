# Assuming subhalo_mass is already defined
# Multiply subhalo_mass by the factor
subhalo_mass_corrected = subhalo_mass * 1e10 / 0.6774

# Filter halos with mass higher than 1e9
mass_threshold = 1e9
filtered_subhalo_mass = subhalo_mass_corrected[subhalo_mass_corrected > mass_threshold]

# Calculate the logarithm of the filtered masses
log_filtered_subhalo_mass = np.log10(filtered_subhalo_mass)

# Define the bin edges with a size of 0.5 in log10
bin_edges = np.arange(np.floor(log_filtered_subhalo_mass.min()), np.ceil(log_filtered_subhalo_mass.max()) + 0.5, 0.5)

# Create a figure and axes for plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Draw a histogram for the logarithm of the filtered subhalo_mass
ax.hist(log_filtered_subhalo_mass, bins=bin_edges, color='blue', alpha=0.7, histtype='bar', rwidth=0.95)
ax.set_xlabel('Log10 Subhalo Mass [$\log_{10}(M_\odot)$]')
ax.set_ylabel('Number of Halos')
ax.set_title('Histogram of Log10 Subhalo Mass for Halos with Mass > $10^9 M_\odot$')
ax.set_yscale('log')
ax.grid(True)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

plt.show()