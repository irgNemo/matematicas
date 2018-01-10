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
	extractedData = dropColumnsByHeader(dataset, ['Sujeto','clase']);
	imputeTo(extractedData, 1.7976931348623157e+108); # Se asigno este valor de manera arbitraria para que no marcara un error de validacion por valores muy grandes
	sujetos = extractColumnsByHeader(dataset, ['Sujeto']);
	# Clustering
	kmeans = KMeans(n_clusters=2, n_init=300, max_iter=500)
	cluster_labels = kmeans.fit_predict(extractedData);
	silhouette_scores = metrics.silhouette_samples(extractedData, cluster_labels, metric='euclidean');
	silhouette_scores = np.reshape(silhouette_scores, (22,1));
	results = np.stack((sujetos,silhouette_scores),axis=-1);
	print(results);

def imputeTo(data, newValue):
	""" 
	Impute the NaN values in the 'data' matrix, with the value specified in 'newValue'
	"""
	data[np.isnan(data)] = newValue; # Se asigno este valor de manera arbitraria para que no marcara un error de validacion por valores muy grandes
	
def extractColumnsByHeader(dataset, headerNameToExtract):
	headers = dataset.header;
	data = dataset.data;
	indexesToExtract = list();
	for toExtract in headerNameToExtract:
		indexesToExtract.extend(np.where(headers == toExtract)[0]);
	data = data[..., np.array(indexesToExtract)]; # Remover columnas con valores categoricos
	return data;

def dropColumnsByHeader(dataset, headerNameToRemove):
	"""
	Extract a sub matrix from the one in dataset but with columns defined in headerNameToRemove
	param dataset: a wrapper containing the data and header 
	param headerNameToRemove: array containing the headers corresponding to the column to be removed
	"""
	headers = dataset.header;
	data = dataset.data;
	indexesToRemove = list();
	for toRemove in headerNameToRemove:
		indexesToRemove.extend(np.where(headers == toRemove)[0]);
	data = np.delete(data, indexesToRemove, 1);
	data = np.asarray(data, dtype=float); # Cambiar el tipo de dato del arreglo a flotante
	return data;

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


	# Remover columnas y convertir arreglo en tipo float
	#columns = data.shape[1] - 2; # Numero de columnas menos las dos ultimas con valores categoricos
	#data = data[..., 0:columns]; # Remover columnas con valores categoricos
	#header = dataset.header;
