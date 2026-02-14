import cv2

class Sharpen:
    def apply(self, image):
        blur = cv2.GaussianBlur(image, (5,5), 0)
        return cv2.addWeighted(image, 1.5, blur, -0.5, 0)
