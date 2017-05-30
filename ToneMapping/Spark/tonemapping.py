import cv2
import numpy as np

def gamma_correction(value, gamma, f_stop):
	return pow((value*pow(2,f_stop)),(1.0/gamma));

img = cv2.imread('images/test8.exr', -1)
test_img = img.reshape(img.shape[0] * img.shape[1], 3)
hdr = sc.parallelize(test_img.tolist())
ldr = hdr.map(lambda pixel: [gamma_correction(val, 1.2, 0.4) for val in pixel]).collect()
result = np.array(ldr, dtype=np.float32).reshape(img.shape[0], img.shape[1], 3)
cv2.imwrite('result_spark.png', result*255)