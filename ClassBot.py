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

        if maxval >= threshold:
            topleft = maxloc

            bottomright = (
                topleft[0] + self.temp_img.shape[0],
                topleft[1] + self.temp_img.shape[1],
            )
            cv.rectangle(
                self.main_img,
                topleft,
                bottomright,
                color=(255, 100, 11),
                thickness=2,
                lineType=cv.LINE_4,
            )

            font = cv.FONT_ITALIC
            position = (topleft[0], topleft[1] + -5)
            fontsize = 1
            color = (255, 100, 11)
            cv.putText(
                self.main_img, "Test", position, font, fontsize, color, thickness=2
            )

            cv.imshow("result", self.main_img)
            cv.waitKey()
            cv.destroyAllWindows()


myBot = ClassBot("images/main_image.png", "images/temp.png")

myBot.search()
