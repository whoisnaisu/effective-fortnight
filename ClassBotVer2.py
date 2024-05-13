import cv2 as cv
import numpy as np


class ClassBot:
    def __init__(self, main_img, temp_img):
        self.main_img = cv.imread(main_img, cv.IMREAD_ANYCOLOR)
        self.temp_img = temp_img = cv.imread(temp_img, cv.IMREAD_ANYCOLOR)

    def search(self):
        result = cv.matchTemplate(self.main_img, self.temp_img, cv.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv.minMaxLoc(result)

        threshold = 0.9
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))

        height = self.temp_img.shape[0]
        width = self.temp_img.shape[1]

        rectangles = []

        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), width, height]
            rectangles.append(rect)
            rectangles.append(rect)

        rectangles, _ = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

        print(rectangles)

        if len(rectangles):
            for x, y, w, h in rectangles:
                topleft = (x, y)
                bottomright = (x + w, y + h)
                cv.rectangle(
                    self.main_img,
                    topleft,
                    bottomright,
                    color=(100, 255, 11),
                    thickness=2,
                    lineType=cv.LINE_4,
                )

        cv.imshow("result", self.main_img)
        cv.waitKey()
        cv.destroyAllWindows()


myBot = ClassBot("images/coins.png", "images/coin.png")
myBot.search()
