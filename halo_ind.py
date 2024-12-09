# label all pixels in the halo range
from scipy.spatial import cKDTree

# Assuming f, subhalo_posx, subhalo_posy, subhalo_radhm, subhalo_vz, subhalo_vmax, z_halo, lya, c, and ray_z are already defined
LOS_xy = np.array((f['ray_pos'][:, 0], f['ray_pos'][:, 1])).T
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

# Create the LOS_MASK array
LOS_MASK = np.zeros((len(f['ray_pos']), len(ray_z)), dtype=np.float16)

# Iterate over each halo
for i, (pos, radius) in enumerate(zip(subhalo_positions, subhalo_radii)):
    # Find all LOS within the radius in the x-y plane
    los_indices0 = tree.query_ball_point(pos, radius)

    # Store the results
    los_halo_ind.append(los_indices0)

    # Calculate the wavelength range for the halo in the z-direction
    halo_dwl_min = (1 + z_halo_cond[i] + subhalo_vz_cond[i] / c - subhalo_vmax_cond[i] / c) * lya
    halo_dwl_max = (1 + z_halo_cond[i] + subhalo_vz_cond[i] / c + subhalo_vmax_cond[i] / c) * lya

    # Find the wavelength indices that are covered by the halo
    wl_indices0 = np.where((ray_z >= halo_dwl_min) & (ray_z <= halo_dwl_max))[0]

    # if i <= 30:
    #    print(z_halo_cond[i],halo_dwl_min,halo_dwl_max,2*subhalo_radhm[condition2][i],len(wl_indices0), len(los_indices0),los_indices0[0:10])

    # the halo wl for each los that intersect with halo i
    wl_los_halo_indices = []
    for j in los_indices0:
        # Store the results
        wl_los_halo_indices.append(wl_indices0)
        LOS_MASK[j, wl_indices0] = 0

    wl_halo_ind_all.append(wl_los_halo_indices)

# los_indices now contains the indices of LOS in the x-y plane for each halo
# wavelength_indices contains the wavelength indices covered by each halo in the z-direction