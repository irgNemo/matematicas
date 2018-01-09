#!/usr/bin/env python3

from sklearn import datasets;
from sklearn.cluster import KMeans;
from sklearn.preprocessing import Imputer
from sklearn import metrics;
import pandas as pd;
import numpy as np;

def main():

	dataset = readDataSet("D_C_all_AM.csv");
	data = dataset.data;
	obtainDataMatrix(dataset,['Sujeto','clase']);

		
	# Impute
	#data[np.isnan(data)] = 1.7976931348623157e+108; # Se asigno este valor de manera arbitraria para que no marcara un error de validacion por valores muy grandes
	#data = np.nan_to_num(data);
	
	# Clustering
	#kmeans = KMeans(n_clusters=3, random_state=1).fit(data);
	#labels = kmeans.labels_;
	#silhouette_scores = metrics.silhouette_samples(data, labels, metric='euclidean');


def obtainDataMatrix(dataset, headerNameToRemove):
	headers = dataset.header;
	data = dataset.data;
	arrayToRemove = list();
	for toRemove in headerNameToRemove:
		arrayToRemove.append(np.where(headers == toRemove)[0]);
	print(arrayToRemove);
	#for array in arrayToRemove:
	#	data = np.delete(data, array);
	#print(indexToRemove);

		

	# Remover columnas y convertir arreglo en tipo float
	#columns = data.shape[1] - 2; # Numero de columnas menos las dos ultimas con valores categoricos
	#data = data[..., 0:columns]; # Remover columnas con valores categoricos
	#data = np.asarray(data, dtype=float); # Cambiar el tipo de dato del arreglo a flotante
	#header = dataset.header;
	

def readDataSet(filename):
	#TODO Validar que filename tenga algo; ademas agregar excepciones
	dataset = pd.read_csv(filename);
	data = dataset.values;
	header = np.asarray(list(dataset));
	return dataWrapper(data, header);
	

class dataWrapper:
	def __init__(self, data, header):
		self.data = data;
		self.header = header;



if __name__ == "__main__":
	main();
