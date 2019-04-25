import numpy as np
import cv2

img = cv2.imread("./pic/hammer.png", cv2.IMREAD_UNCHANGED)
img: np.ndarray = cv2.pyrDown(img)

gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # 找到边界框坐标 find bounding box coordinates
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # find minimum area
    rect = cv2.minAreaRect(c)
    # calculate coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    # print(box)
    # normal coordinates to integers
    box = np.int_(box)
    # draw contours
    try:
        # 这个地方对box的整数类型有要求须int32往上
        # print("Success to dray a box:", box)
        # box = np.array([[22, 33], [24, 33], [77, 33], [55, 33]], dtype=np.int64)
        # print(box.max())
        cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
    except Exception as e:
        print(e.__str__(), box)
        pass

    # calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # cast to integer
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    img = cv2.circle(img, center, radius, (255, 0, 0), 2)

# cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
cv2.imshow("contours", img)
cv2.waitKey(-1)


