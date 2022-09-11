import numpy as np

def fs_ctr_to_aff_ctr(fs_center, fs_radius):
    fs_center_norm = np.abs(fs_center)
    fs_normalized_ctr = fs_center / fs_center_norm

    zmin = np.tan(np.arctan(fs_center_norm) - fs_radius)
    zmax = np.tan(np.arctan(fs_center_norm) + fs_radius)

    return fs_normalized_ctr * (zmin + zmax) / 2

def aff_ctr_to_fs_ctr(aff_center, aff_radius):
    aff_center_norm = np.abs(aff_center)
    return np.tan((np.arctan(aff_center_norm + aff_radius) +
                   np.arctan(aff_center_norm - aff_radius)) / 2)
