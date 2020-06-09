import vtk
import numpy as np
from transforms3d.affines import decompose
from vtk.util import numpy_support


class VolumeRenderer:
    def __init__(self, volume: np.ndarray, affine: np.ndarray):
        assert volume.ndim == 3
        self.volume = volume
        self.affine = affine

        # self
        # self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        # self.mapper.SetInputData(self.volume)
        # self.prop = vtk.vtkProp3D()

        self.prop = self.create_volume_prop(self.volume, self.affine)
        self.renderer = vtk.vtkRenderer()
        self._setup_renderer()

    def render_test(self):
        """
        Show a vtk window, and
        :return:
        """
        self._setup_renderer()
        self._setup_window()
        self._setup_interactor()

        # Start to render
        self._render_window.Render()
        self._render_window_interactor.Start()

    def _setup_renderer(self):
        """
        Setup the renderer, background color and actors
        :return:
        """
        # Same as the 3DSlicer
        self.renderer.SetBackground(0.756863, 0.764706, 0.909804)
        self.renderer.AddActor(self.prop)

    def _setup_interactor(self):
        """
        Setup the interactor, just for unit test
        :return:
        """
        self._render_window_interactor = vtk.vtkRenderWindowInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        # style = vtk.vtkInteractorStyleMultiTouchCamera()
        self._render_window_interactor.SetRenderWindow(self._render_window)
        self._render_window_interactor.SetInteractorStyle(style)
        self._render_window_interactor.Initialize()

    def _setup_window(self):
        """
        Setup render window, just for unit test
        :return:
        """
        self._render_window = vtk.vtkRenderWindow()
        self._render_window.AddRenderer(self.renderer)

    @staticmethod
    def create_volume_prop(volume: np.ndarray, affine: np.ndarray) -> vtk.vtkVolume:
        """
        Convert a numpy 3D matrix to a vtkVolume object
        :param volume:
        :param affine:
        :return:
        """
        volume = volume / volume.max() * 255
        volume = volume.astype('uint8')
        dims = volume.shape

        maxValue = volume.max()
        minValue = volume.min()

        dataImporter = vtk.vtkImageImport()
        data_string = volume.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)

        # todo ?????????????????
        # set data spacing
        _, _, spacing, _ = decompose(affine)
        z = abs(spacing)
        dataImporter.SetDataSpacing(z[2], z[0], z[1])
        # dataImporter.SetDataSpacing(2.4, 0.7, 0.7)

        dataImporter.SetDataExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)
        dataImporter.SetWholeExtent(0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)

        try:
            # Compute by GPU
            volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        except Exception as e:
            logging.warning("Failed to connect to GPU, render with CPU")
            # Compute by CPU
            volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()

        out_port: vtk.vtkAlgorithmOutput = dataImporter.GetOutputPort()
        volume_mapper.SetInputConnection(out_port)

        # The color transfer function maps voxel intensities to colors.
        # It is modality-specific, and often anatomy-specific as well.
        # The goal is to one color for flesh (between 500 and 1000)
        # and another color for bone (1150 and over).
        colorLUT = np.arange(minValue, maxValue, maxValue / 5.0)
        print(colorLUT)
        volumeColor = vtk.vtkColorTransferFunction()
        volumeColor.AddRGBPoint(colorLUT[0], 0.0, 0.0, 0.0)
        volumeColor.AddRGBPoint(colorLUT[1], 1.0, 0.5, 0.3)
        volumeColor.AddRGBPoint(colorLUT[2], 1.0, 0.5, 0.3)
        volumeColor.AddRGBPoint(colorLUT[3], 1.0, 1.0, 0.9)

        # The opacity transfer function is used to control the opacity
        # of different tissue types.
        volumeScalarOpacity = vtk.vtkPiecewiseFunction()
        volumeScalarOpacity.AddPoint(colorLUT[0], 0.00)
        volumeScalarOpacity.AddPoint(colorLUT[1], 0.15)
        volumeScalarOpacity.AddPoint(colorLUT[2], 0.5)
        volumeScalarOpacity.AddPoint(colorLUT[3], 0.85)

        # The gradient opacity function is used to decrease the opacity
        # in the "flat" regions of the volume while maintaining the opacity
        # at the boundaries between tissue types. The gradient is measured
        # as the amount by which the intensity changes over unit distance.
        # For most medical data, the unit distance is 1mm.
        volumeGradientOpacity = vtk.vtkPiecewiseFunction()
        volumeGradientOpacity.AddPoint(0, 0.0)
        volumeGradientOpacity.AddPoint(colorLUT[1] / 2, 0.5)
        volumeGradientOpacity.AddPoint(colorLUT[2] / 2, 1.0)

        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(volumeColor)
        volumeProperty.SetScalarOpacity(volumeScalarOpacity)
        volumeProperty.SetGradientOpacity(volumeGradientOpacity)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        volumeProperty.SetAmbient(0.4)
        volumeProperty.SetDiffuse(0.6)
        volumeProperty.SetSpecular(0.2)

        # The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
        # and orientation of the volume in world coordinates.
        actorVolume = vtk.vtkVolume()
        actorVolume.SetMapper(volume_mapper)
        actorVolume.SetProperty(volumeProperty)
        return actorVolume


if __name__ == '__main__':
    import os
    import logging
    import tools

    logging.basicConfig(level=logging.DEBUG,
                        # filename="debug.log",
                        format="%(asctime)s %(filename)s %(lineno)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        filemode="w")
    # image_item = ImageItem(LoadType.NII, os.path.abspath(
    #     "/home/xu/.xying/nii_for_dcm/0000561924/1.2.528.1.1008.90592741798.1476669119.1210.2"
    #     "/1_2_392_200036_9116_2_5_1_37_2420749461_1476701528_546856.nii.gz"))
    # print(image_item.Shape)
    image = tools.get_arr_from_nii('../master-interview/data/lung.nii')
    renderer = VolumeRenderer(image, np.eye(4))
    renderer.render_test()