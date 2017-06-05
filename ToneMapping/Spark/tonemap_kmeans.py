import sys
import cv2
import numpy as np
from numpy import array
from pyspark.mllib.clustering import KMeans, KMeansModel

def change_image_extension(img):
	return img.replace('.exr', '.png')

def rgb2Lum(pixel):
	return pixel[0] * 0.0722 + pixel[1] * 0.7152 + pixel[2] * 0.2126

def getLum(bgr):
	Lum = []
	for pixel in bgr:
		Lum += [rgb2Lum(pixel)]
	return np.array(Lum, dtype=np.float32)

def search_hist(hist, value):
	for item in hist:
		if item[0] == value:
			return item[1]

def search_mean(means, val):
	for item in means:
		if item[0] == val:
			return item[1]

def weighted_mean(hist, point): #point = (cluster_idx, [points])
	mean = sum([(search_hist(hist, pixel) * pixel) for pixel in point[1]]) / sum([search_hist(hist, pixel) for pixel in point[1]])
	return mean

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: tonemap_kmeans <file>")
	        exit(-1)

	from pyspark import SparkContext

	file_name = sys.argv[1]

	sc = SparkContext(appName="ToneMapping_Kmeans")
	
	img = cv2.imread(file_name, -1)
	bgr = img.reshape(img.shape[0] * img.shape[1], 3)
	bgr = sc.parallelize(bgr.tolist()).cache()

	L = np.array(bgr.map(lambda pixel: [rgb2Lum(pixel)]).collect(), dtype=np.float32)

	rdd = sc.parallelize(L.tolist()).cache()
	hist = rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda x,y: x+y).collect()

	clusters = KMeans.train(rdd, 256, initializationMode="random", maxIterations=1000)

	prediction = rdd.map(lambda x: (clusters.predict(x), x)).reduceByKey(lambda x, y: x+y)
	weighted_means = prediction.map(lambda x: (x[0], weighted_mean(hist, x)))
	means = weighted_means.collect()

	nL = rdd.map(lambda L: [search_mean(means, clusters.predict(L))])
	nnL = np.array(nL.collect(), dtype=np.float32)
	scale = nnL / L

	ldr = np.array(bgr.collect(), dtype=np.float32)
	ldr *= scale

	result = ldr.reshape(img.shape[0], img.shape[1], 3)
	cv2.imwrite(change_image_extension(file_name), result*255)

	sc.stop()
