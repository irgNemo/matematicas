#!/usr/bin/env python3

from sklearn import datasets;
from sklearn.cluster import KMeans;
from sklearn.preprocessing import Imputer
from sklearn import metrics;
import pandas as pd;
import numpy as np;
from pathlib import Path;
import re;

def main():

	tiempos = ['Durante','Despues'];
	generos = ['H','M'];
	bandas = ['sf','a','b','d','g','t'];
	metricas = ['C','E','Gio'];
	clases = ['AM', 'MB'];
	directoryPath = "./Datasets/";
	
	outliers = computeOutliersPerTimeGenderBand(tiempos, generos, bandas, metricas, clases, directoryPath, 110);
	print(outliers);
	#outliersUnionPerBand(outliers);


def outliersUnionPerBand(outliersPerBand):
	
	unionPerBand = dict();
	for band in outliersPerBand:
		unionPerMetric = dict();
		for metric in outliersPerBand[band]:
			unionPerMetric[metric] = None;
			for clase in outliersPerBand[band][metric]:
				#print(outliersPerBand[band][metric][clase]);
				sujetos = outliersPerBand[band][metric][clase][..., 2]; # TODO Ver si los seleccionamos por cabecera
				if unionPerMetric[metric] is None:
					unionPerMetric[metric] = sujetos;
				else:
					unionPerMetric[metric] = np.union1d(unionPerMetric[metric], sujetos);
		unionPerBand[band] = unionPerMetric;
	print(unionPerBand);

def computeOutliersPerTimeGenderBand(times, genders, bands, metrics, clases, directoryPath, outliersThreshold):
# TODO Pasar las cabeceras de metadatos
	#outliersPerBand = dict();
	outlier = dict();
	for time in times:
		outlier[time] = dict();
		for gender in genders:
			outlier[time][gender] = dict();
			for band in bands:
				outlier[time][gender][band] = dict();
				outliersPerMetric = dict();
				for metric in metrics:
					outlier[time][gender][band][metric] = dict();
					outliersPerClass = dict();
					for clase in clases:
						filename = time + "_" + metric + "_" + band + "_" + gender + "_" + clase;
						completePathFile = directoryPath + filename + ".csv";
						my_file = Path(completePathFile);
						classMatch= re.search('(.*)M(.*)', clase);
						classToAnalyse = "".join(classMatch.groups());
						
						if not my_file.is_file(): # If the file not exists, continues with the following one
							continue;
						dataset = readDataSet(completePathFile);
						data = dataset.data;
						extractedData = dropColumnsByHeader(dataset, ['Sujeto','Clase']); # Obtenemos la matriz de datos sin los metadatos
						imputeNaN(extractedData, 1.7976931348623157e+108); # Se asigno este valor de manera arbitraria para que no marcara un error de validacion por valores muy grandes
						sujetos = extractColumnsByHeader(dataset, ['Sujeto']);# Se obtiene una lista de los sujetos
						clasesnarray = extractColumnsByHeader(dataset, ['Clase']);
						sujetos_menores_cero = detectOutliers(extractedData, clasesnarray, sujetos, 'euclidean', 0, '<', 1);
						sujetos_sin_medios = removeRowsByColumnValues(sujetos_menores_cero, 'eq', 0, 'M');
						
						rowPerClass = countRowsWhere(clasesnarray, 0, classToAnalyse);
						rowPerOutliers = countRowsWhere(sujetos_sin_medios, 0, classToAnalyse);
						
						percentageOutliers = (rowPerOutliers/rowPerClass)*100;
						
						if percentageOutliers < outliersThreshold:
							outliersPerClass[classToAnalyse] = sujetos_sin_medios;

					if outliersPerClass:
						#outliersPerMetric[metric] = outliersPerClass;
						outlier[time][gender][band][metric][classToAnalyse] = outliersPerClass;
			
				if outliersPerMetric:
					outlier[time][gender][band] = outliersPerMetric;
	return outlier;

				

def countRowsWhere(data, columnToAnylise, searchValue):
	indexes = np.argwhere(data[..., columnToAnylise] == searchValue);
	return indexes.flatten().size;



def detectOutliers(data, clases, sujetos, metric, threshold, condition, columnToAnalyse):
	"""
	Detect subject with a silhouette score less than threshold
	"""
	silhouette_scores = cohesionBySilhouette(data, clases.flatten(), metric); # Se obtiene los scores de silhouette para el conjunto de datos utilizando la etiquetas de clase como sus clusters 
	silhouette_clase = np.append(clases, silhouette_scores, axis=1); # Se crea una matriz que contiene a los sujetos y sus score de silhouettea
	silhouette_clases_sujetos = np.append(silhouette_clase, sujetos, axis=1);
	sujetos_silhouette_menor_cero = removeRowsByColumnValues(silhouette_clases_sujetos, condition, columnToAnalyse, threshold);
	return sujetos_silhouette_menor_cero;
	

def validateLessEqualPercentage(data, subdata, threshold):
	"""
	Validate
	return: True or False
	"""
	row_data_size = data.shape[0];
	row_subdata_size = subdata.shape[0];

	return True if (((row_subdata_size*100)/row_data_size) <= threshold) else False;


def removeRowsByColumnValues(data, condition, columnToAnalyse, threshold):
	"""
	Filter
	"""
	indexes = None;
	if condition == '<':
		indexes = np.argwhere(data[..., columnToAnalyse] >= threshold);
		indexes = indexes.flatten();
	elif condition == 'eq':
		indexes = np.argwhere(data[..., columnToAnalyse] == threshold);
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
	rows = data.shape[0];
	silhouette_scores = metrics.silhouette_samples(data, clases, metric);
	silhouette_scores = np.reshape(silhouette_scores, (rows,1));
	return silhouette_scores;	

def clusteringAndSilhouette(data, n_clusters, n_init, max_iter, metric='euclidean'):
	rows = data.shape[0];
	kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter);
	cluster_labels = kmeans.fit_predict(data);
	silhouette_scores = metrics.silhouette_samples(data, cluster_labels, metric);
	silhouette_scores = np.reshape(silhouette_scores, (rows,1));
	cluster_labels = np.reshape(cluster_labels, (rows,1));
	silhouette_scores = np.append(silhouette_scores, cluster_labels, axis=1);
	return	silhouette_scores;

def imputeNaN(data, newValue):
	""" 
	Impute the NaN values in the 'data' matrix, with the value specified in 'newValue'
	"""
	data[np.isnan(data)] = newValue; # Se asigno este valor de manera arbitraria para que no marcara un error de validacion por valores muy grandes
	
def extractColumnsByIndex(data, index):
	data = data[..., index]; # Remover columnas con valores categoricos
	return data;

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


	"""
	tuples = filenamesTuples(tiempos, generos, bandas, metricas, clases);
	for pair in tuples:
		for filename in pair:
			completePathFile = directoryPath + filename;
			my_file = Path(completePathFile);
			if not my_file.is_file():
				#print("El archivo " + filename  +  " no fue encontrado");
				continue;
			print(completePathFile);
			dataset = readDataSet(completePathFile);
			data = dataset.data;
			extractedData = dropColumnsByHeader(dataset, ['Sujeto','Clase']); # Obtenemos la matriz de datos sin los metadatos
			imputeNaN(extractedData, 1.7976931348623157e+108); # Se asigno este valor de manera arbitraria para que no marcara un error de validacion por valores muy grandes
			sujetos = extractColumnsByHeader(dataset, ['Sujeto']);# Se obtiene una lista de los sujetos
			clases = extractColumnsByHeader(dataset, ['Clase']);
			sujetos_menores_cero = detectOutliers(extractedData, clases, sujetos, 'euclidean', 0, 'lt', 1);
			print(sujetos_menores_cero);
			#if not validateLessEqualPercentage(extractedData, sujetos_menores_cero, 40):
			#	print("Outlayers mayores al porcentaje en archivo: " + filename);	
	"""
