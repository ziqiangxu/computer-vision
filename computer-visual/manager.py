import cv2
import time
import numpy as np


class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFileName = None
        self._videoFileName = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def isWritingImage(self):
        if self._imageFileName:
            return True
        return False

    @property
    def isWritingVideo(self):
        if self._videoFileName:
            return True
        return False

    def enterFrame(self):
        assert not self._enteredFrame, '检查先前的frame是否退出'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """
        绘制到窗口， 写到文件， 释放frame
        """
        # 检查是否有可以获取的frame
        if self.frame is None:
            self.enterFrame = False
            return
        # 更新FPS估计值和相关变量
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            # 这一步有问题啊???看书
            self._fpsEstimate = self._framesElapsed / timeElapsed  # 每秒钟多少frame
        self._framesElapsed += 1

        # 有则绘制到窗口
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)
        # 写到image文件
        if self.isWritingImage:
            cv2.imwrite(self._imageFileName, self._frame)
            self._imageFileName = None

        # 写到video文件
        self._writeVideoFram()

        # 释放frame
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        self._imageFileName = filename

    def startWritingVideo(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        self._videoFileName = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        self._videoFileName = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps < 1:
                # fps未知，则进行估计
                if self._framesElapsed < 20:
                    # 等待更多帧数，让估计更加准确
                    return
                else:
                    fps = self._fpsEstimate
            size = (
                int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

            # 此处的缩进问题？？？？
            self._videoWriter = cv2.VideoWriter(
                self._videoFileName, self._videoEncoding, fps, size
            )
            self._videoWriter.write(self._frame)
        # self._videoWriter = cv2.VideoWriter(
        #     self._videoFileName, self._videoEncoding, fps, size
        # )
        # self._videoWriter.write(self._frame)


class WindowManager(object):
    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def show(self, frame):
        print("frame", frame)
        cv2.imshow(self._windowName, frame)

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            keycode &= 0xFF
            self.keypressCallback(keycode)