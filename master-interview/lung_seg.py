import torch # must be first line, avoid dlopen error. Daryl Xu
from lungmask import mask
import SimpleITK as sitk
import numpy as np
import tools


def get_lung_atlas(image_path: str) -> np.ndarray:
    """
    Get lung area by https://github.com/JoHof/lungmask.git, combine the output of all models
    todo Just run a model
    :param image_path image which supports by SimpleITK
    :return:
    """
    input_image = sitk.ReadImage(image_path)
    # Get models
    # model_r231 = mask.get_model('unet', 'R231')
    model_lobes = mask.get_model('unet', 'LTRCLobes')
    # model_covi = mask.get_model('unet', 'R231CovidWeb')

    # segmentation_r231 = mask.apply(input_image)  # default model is U-net(R231)
    # tools.npy2nii(segmentation_r231, 'data/log/lung-entity-r231.nii')
    #
    # segmentation_lobes = mask.apply(input_image, model_lobes)
    # tools.npy2nii(segmentation_lobes, 'data/log/lung-entity-lobes.nii')

    segmentation_lobes_r231 = mask.apply_fused(input_image)
    tools.npy2nii(segmentation_lobes_r231, 'data/log/lung-entity-lobes-r231.nii')

    # segmentation_covi = mask.apply(model_covi)
    # tools.npy2nii(segmentation_covi, 'data/lung-entity-covi.nii')

    segmentation_combine = segmentation_lobes_r231
    # segmentation_combine = segmentation_lobes_r231 + segmentation_lobes + segmentation_r231
    tools.npy2nii(segmentation_combine, 'data/log/lung-entity-combine.nii')
    return segmentation_combine


def get_roi(image: np.ndarray, atlas: np.ndarray) -> np.ndarray:
    """
    Get the image specific by atlas
    :param image:
    :param atlas:
    :return:
    """
    atlas[atlas != 0] = 1
    return image * atlas


def my_get_lung_atlas():
    # first chanel as slice axis, best performance
    lung_atlas = get_lung_atlas("data/lung-trans-201.vtk")
    tools.npy2nii(lung_atlas, 'data/output/lung-atlas.nii')


def my_get_lung_roi():
    # Get lung roi
    atlas = tools.get_arr_from_nii('data/log/lung-entity-combine.nii')
    image = tools.get_arr_from_nii('data/lung-trans-201.nii')
    lung_roi = get_roi(image, atlas)
    tools.npy2nii(lung_roi, 'data/output/lung-roi.nii')


if __name__ == '__main__':
    my_get_lung_atlas()
    my_get_lung_roi()
