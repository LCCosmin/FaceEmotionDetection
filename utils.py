import cv2
def normalize(img, width, height):
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (1,1), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #img = img / 255
    #img = cv2.Canny(img, 10, 255)

    return img