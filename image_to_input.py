import numpy as np
from osgeo import gdal

def get_input_data(filename):
	ds = gdal.Open(filename)

	myarray = np.array(ds.ReadAsArray())
	b, x, y = myarray.shape
	n_bands = 7
	counter = 0
	band_values = {}
	for i in range(x):
		for j in range(y):
			for v in myarray[:,i,j]:
				if counter in band_values:
					band_values[counter].append(v)
				else:
					band_values[counter] = [v]
				counter += 1
				if counter == n_bands:
					counter = 0

	histogram = []
	n_bins = 50
	for i in range(7):
		histogram_raw = np.histogram(band_values[i], bins=n_bins)
		histogram_processed = []
		total = 0.0
		for j in range(n_bins):
			individual = histogram_raw[0][j] * (histogram_raw[1][j] + histogram_raw[1][j+1])/2.0
			total += individual
			histogram_processed.append(individual)

		for j in histogram_processed:
			histogram.append(j/total)

		#histogram.append(total/n_bins)

	return histogram

if __name__ == '__main__':
	get_input_data('image_data/1071001_0-0_648.tif')