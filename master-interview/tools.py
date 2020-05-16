import numpy as np
import nibabel as nib
# my cv version(role back when the project end), 3.4.2.17


def npy2nii(volume: np.ndarray, nii_path: str):
    """
    convert npy to nii
    :param volume:
    :param nii_path:
    :return:
    """
    # because of nibabel, the saving path can't contain more than one dot(.)
    assert nii_path.endswith('.nii')
    assert nii_path.count('.') == 1
    affine = np.eye(4)
    affine[0, 0] = 1
    img = nib.nifti1.Nifti1Image(volume, affine)
    # img.set_data_dtype(np.int32)
    try:
        nib.nifti1.save(img, nii_path)
    except:
        print("failed to convert npy to nii")


def get_arr_from_nii(nii_path: str) -> np.ndarray:
    """
    Get data from NIfTI
    :param nii_path:
    :return:
    """
    nii = nib.load(nii_path)
    return nii.get_data()
