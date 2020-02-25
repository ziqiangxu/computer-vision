import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show_img(img: np.ndarray):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


class ObjectDetector:
    MIN_MATCH_COUNT = 10
    sift = cv.xfeatures2d.SIFT_create()

    def __init__(self, query_img_path: str):
        self.query_img: np.ndarray = cv.imread(query_img_path)
        # 进行预处理，边缘检测

        self.query_img_gray: np.ndarray = cv.cvtColor(self.query_img, cv.COLOR_BGR2GRAY)

        self.kp, self.des_query = self.sift.detectAndCompute(self.query_img_gray, None)

        h, w, d = self.query_img.shape
        # 取得query的四个角
        self.pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape([-1, 1, 2])

        cv.drawKeypoints(self.query_img, self.kp, self.query_img)

    def detect(self, target_img_path: str, output_log=False):
        target_img = cv.imread(target_img_path)
        # target_img = cv.resize(target_img, (800, 600))
        target_img_gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)

        kp_target, des_target = self.sift.detectAndCompute(target_img_gray, None)

        # todo 比对特征点
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.des_query, des_target, 2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # todo connect the keypoints
        if len(good) < self.MIN_MATCH_COUNT:
            raise Exception(f'Not enough matches are found: {len(good)}/{self.MIN_MATCH_COUNT}')
        print(f'matches are found: {len(good)}/{self.MIN_MATCH_COUNT}')

        src_pts = np.float32([self.kp[m.queryIdx].pt for m in good]).reshape([-1, 1, 2])
        # points 的shape
        target_pts = np.float32([kp_target[m.trainIdx].pt for m in good]).reshape([-1, 1, 2])

        # todo RANSAC 算法
        # trans_matrix, mask = cv.findHomography(src_pts, target_pts, cv.RANSAC, 5.0)
        trans_matrix, mask = cv.findHomography(src_pts, target_pts, cv.RANSAC, 5, confidence=0.997)

        matches_mask = mask.ravel().tolist()

        dst = cv.perspectiveTransform(self.pts, trans_matrix)

        result_img = np.copy(target_img)
        result_img = cv.polylines(result_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

        draw_params = dict(matchColor=(0, 255, 0),  # in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # 只连接正常值 inliers
                           flags=2)
        cv.drawKeypoints(target_img, kp_target, target_img)
        img_inliers = cv.drawMatches(self.query_img, self.kp, target_img, kp_target, good, None, **draw_params)
        # plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))

        if output_log:
            cv.imwrite('log/query-key-points.png', cv.drawKeypoints(self.query_img, self.kp, self.query_img))
            cv.imwrite('log/target-key-points.png', cv.drawKeypoints(target_img, kp_target, None))
            cv.imwrite('log/result.png', result_img)
            cv.imwrite('log/inliers.png', img_inliers)
        # plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
        # plt.show()
        # target_img = cv.drawKeypoints(target_img_gray, good, target_img)
        # cv.imwrite('target-out.img', target_img)
        return result_img


if __name__ == '__main__':
    # 汽车，query从target
    # object_detector = ObjectDetector('query1.png')
    # res_img = object_detector.detect('target1.jpeg', True)
    # show_img(res_img)

    # 温度计 手机拍摄 效果不好
    # object_detector = ObjectDetector('query2-camera.jpg')
    # res_img = object_detector.detect('target2.jpeg', True)
    # show_img(res_img)

    # 板凳， query为从target图像中截取并经过仿射变换
    # object_detector = ObjectDetector('query3.png')
    # res_img = object_detector.detect('target3.jpeg', True)
    # show_img(res_img)

    object_detector = ObjectDetector('query3-camera.jpeg')
    res_img = object_detector.detect('target3.jpeg', True)
    show_img(res_img)

    # 手写文字， query为手机拍摄
    # object_detector = ObjectDetector('query4-camera.jpeg')
    # res_img = object_detector.detect('target4.jpg', True)
    # show_img(res_img)

    # 对于查询图片，在以下条件下检测效果好：有复杂纹理，图片由严格的仿射变换获得。
