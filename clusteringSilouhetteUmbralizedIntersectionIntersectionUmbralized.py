#!/usr/bin/env python3

from sklearn import datasets;
from sklearn.cluster import KMeans;
from sklearn.preprocessing import Imputer
from sklearn import metrics;
import pandas as pd;
import numpy as np;
from pathlib import Path;
import re;

# Revisiones antes de ejecutar
# 1.- La estructura de los nombres de los archivos en donde se construye el nombre linea 80 
# 2.- Los nombres de los tiempos, generos, bandas, metricas, clases que estan en el main. Linea 18 - 22
# 3.- Revisar el nombre del directorio contenedor de los conjuntos de datos (Datasets). Cambiar si es necesario.
# 4.- Cambiar los valores de las cabeceras de los archivos. E.j "Sujeto" y "Clase" para que coincidan con el del contenido del archivo. TODO Corregir esto urgentemente


def main():

	tiempos = ['durante','despues'];
	generos = ['H','M','HM'];
	bandas = ['sf','alpha','beta','gamma','delta','theta'];
	metricas = ['C','E','Gio'];
	clases = ['AMB'];
	directoryPath = "./Datasets/Datasets_Rendimiento_conOutliers/";
	
	outliers = computeOutliersPerTimeGenderBand(tiempos, generos, bandas, metricas, clases, directoryPath, 30);
	intersection = outliersIntersectionPerMetric(outliers);
	intersection = outliersIntersectionPerBand(intersection, 2);
	

def outliersReport(outliers, outlierUnion):
	#print("Generando reporte");
	separador = "-";
	for time in outliers:
		for gender in outliers[time]:
			for band in outliers[time][gender]:
				print("General: " + time + separador + gender + separador + band);
				for metric in outliers[time][gender][band]:
					cabecera = time + separador + gender + separador + band + separador + metric;
					print("\nMetrica: " + metric);
					for clase in outliers[time][gender][band][metric]:
						print("Clase: " + clase);
						print(outliers[time][gender][band][metric][clase][...,2]);
				print("\nUnion: "); 
				print(outlierUnion[time][gender][band]);
				print("---------------------------------------------------------");
				print("\n");
	
def outliersIntersectionPerMetric(outliers):
	#print("Uniendo outliers");
	intersection = dict();
	for time in outliers:
		for gender in outliers[time]:
			for band in outliers[time][gender]:
				for metric in outliers[time][gender][band]:
					for clase in outliers[time][gender][band][metric]:
						sujetos = outliers[time][gender][band][metric][clase][..., 2]; # TODO Ver si los seleccionamos por cabecera en lugar del indice
						if time not in intersection:
							intersection[time] = dict();
						if gender not in intersection[time]:
							intersection[time][gender] = dict();
						if band not in intersection[time][gender]:
							intersection[time][gender][band] = None;
						
						if intersection[time][gender][band] is None:
							intersection[time][gender][band] = sujetos;
						else:
							intersection[time][gender][band] = np.intersect1d(intersection[time][gender][band], sujetos);
							
	return intersection;


def outliersIntersectionPerBand(outliers, threshold):
	intersection = dict();
	separador = "-";
	for time in outliers:
		if time not in intersection:
			intersection[time] = dict();
		for gender in outliers[time]:
			if gender not in intersection[time]:
				intersection[time][gender] = list();
			print("General:" + time + separador + gender);
			numSujetos = dict();
			for band in outliers[time][gender]:
				#print(band);
				sujetos = outliers[time][gender][band];

				for sujeto in sujetos:
					if sujeto not in numSujetos:
						numSujetos[sujeto] = 1;
					else:
						numSujetos[sujeto] += 1; 

			for sujeto in numSujetos:
				if numSujetos[sujeto] >= threshold:
					intersection[time][gender].append(sujeto);

			print(intersection[time][gender]);
	
	return intersection;


def computeOutliersPerTimeGenderBand(times, genders, bands, metrics, clases, directoryPath, outliersThreshold):
# TODO Pasar las cabeceras de metadatos
	#print("Calculando outliers");
	outlier = dict();
	for time in times:
		for gender in genders:
			for band in bands:
				for metric in metrics:
					outliersPerClass = dict();
					for clase in clases:
						filename = time + "_" + metric + "_" + band + "_" + gender + "_" + clase + "_claseRendimiento";
						completePathFile = directoryPath + filename + ".csv";
						my_file = Path(completePathFile);
						
						if not my_file.is_file(): # If the file not exists, continues with the following one
							continue;
						#print(completePathFile); #TODO Quitar
						dataset = readDataSet(completePathFile);
						data = dataset.data;
						extractedData = dropColumnsByHeader(dataset, ['Sujeto','Clase']); # Obtenemos la matriz de datos sin los metadatos
						imputeNaN(extractedData, 1.7976931348623157e+108); # Se asigno este valor de manera arbitraria para que no marcara un error de validacion por valores muy grandes
						sujetos = extractColumnsByHeader(dataset, ['Sujeto']);# Se obtiene una lista de los sujetos
						clasesnarray = extractColumnsByHeader(dataset, ['Clase']);
						sujetos_menores_cero = detectOutliers(extractedData, clasesnarray, sujetos, 'euclidean', 0, '<', 1);

						#print(sujetos_menores_cero); #TODO Quitar
						#print("\n"); #TODO Quitar
						for clase in clase:
							totalClassSize = countRowsWhere(clasesnarray,0,clase);
							outliersClassSize = countRowsWhere(sujetos_menores_cero,0,clase);
							#print("total clase " + clase + ": " + str(totalClassSize));#TODO Quitar
							#print("total outliers " + clase + ": " + str(outliersClassSize) + "\n"); #TODO Quitar
							if (((outliersClassSize/totalClassSize)*100) > outliersThreshold):
								sujetos_menores_cero = removeRowsByColumnValues(sujetos_menores_cero, "eq", 0, clase);
						#print(sujetos_menores_cero); #TODO Quitar
						#print("---------------------------");#TODO Quitar
				
						if (sujetos_menores_cero.size > 0):
							outliersPerClass[clase] = sujetos_menores_cero;

					if outliersPerClass:
						if time not in outlier:
							outlier[time] = dict();
						if gender not in outlier[time]:
							outlier[time][gender] = dict();
						if band not in outlier[time][gender]:
							outlier[time][gender][band] = dict();
						if metric not in outlier[time][gender][band]:
							outlier[time][gender][band][metric] = dict();
						
						outlier[time][gender][band][metric] = outliersPerClass;
	return outlier;

def countRowsWhere(data, columnToAnylize, searchValue):
	indexes = np.argwhere(data[..., columnToAnylize] == searchValue);
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
			#if not validateLessEqualPercentage(extractedData, sujetos_menores_cero, 40):
			#	print("Outlayers mayores al porcentaje en archivo: " + filename);	
	"""
