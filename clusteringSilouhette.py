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
	extractedData = dropColumnsByHeader(dataset, ['Sujeto','clase']); # Obtenemos la matriz de datos sin los metadatos
	imputeNaN(extractedData, 1.7976931348623157e+108); # Se asigno este valor de manera arbitraria para que no marcara un error de validacion por valores muy grandes
	sujetos = extractColumnsByHeader(dataset, ['Sujeto']);# Se obtiene una lista de los sujetos
	clases = extractColumnsByHeader(dataset, ['clase']);
	#clusteringAndSilhouette(extractedData, 2, 300, 500, metric='euclidean');
	silhouette_scores = cohesionBySilhouette(extractedData, clases.flatten(), 'euclidean'); # Se obtiene los scores de silhouette para el conjunto de datos utilizando la etiquetas de clase como sus clusters 
	silhouette_clase = np.append(clases, silhouette_scores, axis=1); # Se crea una matriz que contiene a los sujetos y sus score de silhouettea
	silhouette_clases_sujetos = np.append(silhouette_clase, sujetos, axis=1);
	sujetos_silhouette_menor_cero = filterByColumnValues(silhouette_clases_sujetos, 'lt', 1, 0);
	print(sujetos_silhouette_menor_cero);
	print(validateLessEqualPercentage(extractedData, sujetos_silhouette_menor_cero, 40));


def validateLessEqualPercentage(data, subdata, threshold):
	"""
	Validate
	return: True or False
	"""
	row_data_size = data.shape[0];
	row_subdata_size = subdata.shape[0];

	return True if (((row_subdata_size*100)/row_data_size) <= threshold) else False;


def filterByColumnValues(data, condition, columnToAnalyse, threshold):
	"""
	Filter
	"""
	indexes = None;
	if condition == 'lt':
		indexes = np.argwhere(data[..., columnToAnalyse] >= threshold);
		indexes = indexes.flatten();
	else:
		print('implement condition');

	newData = np.delete(data, indexes, 0);
	return newData;
	

def cohesionBySilhouette(data, clases, metric='euclidean'):
	""" 
	Measure the cohesion of a group of elements tagged by a clas	
	return: Array with silhouette scores and cluster label asigned
	"""
	silhouette_scores = metrics.silhouette_samples(data, clases, metric);
	silhouette_scores = np.reshape(silhouette_scores, (22,1));
	return silhouette_scores;	

def clusteringAndSilhouette(data, n_clusters, n_init, max_iter, metric='euclidean'):
	kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter);
	cluster_labels = kmeans.fit_predict(data);
	silhouette_scores = metrics.silhouette_samples(data, cluster_labels, metric);
	silhouette_scores = np.reshape(silhouette_scores, (22,1));
	cluster_labels = np.reshape(cluster_labels, (22,1));
	silhouette_scores = np.append(silhouette_scores, cluster_labels, axis=1);
	return	silhouette_scores;

def imputeNaN(data, newValue):
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
