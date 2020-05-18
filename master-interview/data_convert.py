import scipy.io as io
import numpy as np
import SimpleITK
import tools


def prepare_data():
    mat = io.loadmat('data/nod.mat')
    nod = mat['nod']
    np.save('data/nod.npy', nod)
    tools.npy2nii(nod, 'data/nod.nii')

    mat = io.loadmat('data/lung.mat')
    lung: np.ndarray = mat['vol']
    np.save('data/lung.npy', lung)
    tools.npy2nii(lung, 'data/lung.nii')

    lung_trans_201 = lung.transpose([2, 0, 1])
    lung_trans_201_itk = SimpleITK.GetImageFromArray(lung_trans_201)
    writer = SimpleITK.ImageFileWriter()
    writer.SetFileName('data/lung-trans-201.vtk')
    writer.Execute(lung_trans_201_itk)
    tools.npy2nii(lung_trans_201, 'data/lung-trans-201.nii')

    lung_itk = SimpleITK.GetImageFromArray(lung)
    writer = SimpleITK.ImageFileWriter()
    writer.SetFileName('data/lung_by_simple_itk.nii')
    writer.Execute(lung_itk)

    nod_itk = SimpleITK.GetImageFromArray(nod)
    writer = SimpleITK.ImageFileWriter()
    writer.SetFileName('data/nod_by_simple_itk.nii')
    writer.Execute(nod_itk)


if __name__ == '__main__':
    prepare_data()
