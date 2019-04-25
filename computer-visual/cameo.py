from manager import WindowManager, CaptureManager
import cv2


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('cameo', self.onKeypress)
        # self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self._captureManager = CaptureManager(
            cv2.VideoCapture("/home/xu/Desktop/深度录屏_选择区域_20181231234930.mp4"),
            self._windowManager
        )

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame
            frame = self._captureManager.frame
            self._windowManager.show(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """
        处理按键事件
        space  截图
        tab    screencast
        escape 退出
        """
        if keycode == 32:  # space
            print("space pressed")
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # tab
            print("tab pressed")
            if self._captureManager.isWritingVideo:
                self._captureManager.stopWritingVideo()
            else:
                self._captureManager.startWritingVideo('screencast.avi')
        elif keycode == 27:  # escape
            print("escape pressed")
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    Cameo().run()
