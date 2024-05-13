import cv2 as cv
import numpy as np

main_img = cv.imread("images/main_image.png", cv.IMREAD_ANYCOLOR)
temp_img = cv.imread("images/temp.png", cv.IMREAD_ANYCOLOR)

result = cv.matchTemplate(main_img, temp_img, cv.TM_CCOEFF_NORMED)

_, maxval, _, maxloc = cv.minMaxLoc(result)

threshold = 0.9

if maxval >= threshold:
    topleft = maxloc

    bottomright = (topleft[0] + temp_img.shape[0], topleft[1] + temp_img.shape[1])
    cv.rectangle(
        main_img,
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
    cv.putText(main_img, "Test", position, font, fontsize, color, thickness=2)

    cv.imshow("result", main_img)
    cv.waitKey()
    cv.destroyAllWindows()
