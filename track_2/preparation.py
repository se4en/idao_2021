import cv2


def prep_img(path):
    new_size = 300
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_den = cv2.fastNlMeansDenoising(img)
    img_crop = img_den[int(img_den.shape[0] / 2 - new_size / 2):int(img_den.shape[0] / 2 + new_size / 2),
               int(img_den.shape[1] / 2 - new_size / 2):int(img_den.shape[1] / 2 + new_size / 2)]
    return img_crop
