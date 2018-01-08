#!/usr/bin/env python3

from sklearn import datasets;
from sklearn.cluster import KMeans;
from sklearn.preprocessing import Imputer
from sklearn import metrics;
import pandas as pd;
import numpy as np;

def main():
	#dataset = datasets.load_iris();
	dataset = readDataSet("D_C_all_AM.csv");
	data = dataset.data;
	columns = data.shape[1] - 2; # Numero de columnas menos las dos ultimas con valores categoricos
	data = data[..., 0:columns]; # Remover columnas con valores categoricos
	target = dataset.target;
	#print(data);
	# Impute
	imp = Imputer(missing_values='?', strategy='mean', axis=0);
	imp.fit(data);
	# Clustering
	#kmeans = KMeans(n_clusters=3, random_state=1).fit(data);
	#labels = kmeans.labels_;
	#silhouette_scores = metrics.silhouette_samples(data, labels, metric='euclidean');
	#print (silhouette_scores);



def readDataSet(filename):
	#TODO Validar que filename tenga algo; ademas agregar excepciones
	dataset = pd.read_csv(filename);
	data = dataset.values;
	target = np.asarray(list(dataset));
	return dataWrapper(data, target);
	

class dataWrapper:
	def __init__(self, data, target):
		self.data = data;
		self.target = target;



if __name__ == "__main__":
	main();
