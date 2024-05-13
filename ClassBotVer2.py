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

        if locations:
            height = self.temp_img.shape[0]
            width = self.temp_img.shape[1]

            for loc in locations:
                bottomright = (loc[0] + width, loc[1] + height)

                cv.rectangle(
                    self.main_img,
                    loc,
                    bottomright,
                    color=(100, 255, 11),
                    thickness=2,
                    lineType=cv.LINE_4,
                )

                font = cv.FONT_ITALIC
                position = (loc[0], loc[1] + -5)
                fontsize = 1
                color = (100, 255, 11)
                cv.putText(
                    self.main_img, "Coin", position, font, fontsize, color, thickness=2
                )

        cv.imshow("result", self.main_img)
        cv.waitKey()
        cv.destroyAllWindows()


myBot = ClassBot("images/coins.png", "images/coin.png")
myBot.search()
