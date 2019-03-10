'''
CS.py
Author: Wenxing Zhang
Date: 03/10/2018
Description: 
Coreset construction using Lucic et al.'s algorithm
(Mario Lucic, Olivier Bachem, and Andreas Krause. 2016. Strong Coresets for Hard
and Soft Bregman Clustering with Applications to Exponential Family Mixtures.
In Artificial Intelligence and Statistics.)
'''

import math
import numpy as np
from scipy.spatial import distance 
import argparse
import sys

		
def ReadInData(filename, attributes, seperator, start_row = 0, start_col = None, end_col = None):
	with open(filename) as datafile:
		for i in range(start_row):
			datafile.readline()

		dataset = []
		for line in datafile:
			data = line.split(seperator)        
			data = data[start_col:end_col]
			if(len(data) != attributes):
				print("The number of attributes specified does not match the data! Exit the program!")
				sys.exit()
			vlist=[]
			for value in data:
				vlist.append(float(value.strip()))
			dataset.append(vlist)
	return dataset

def D2Sampling(dataset, cluster_number, distance_func):
	n = len(dataset)
	B = []
	B.append(dataset[np.random.randint(0,n)]) #i = 0

	for i in range(1, cluster_number): #i = 1,2,...,k-1
		probability = np.zeros(n)
		for k in range(n):
			probability[k] = np.amin(distance.cdist([dataset[k]], B, distance_func),axis=1)[0]
		probability = probability/np.sum(probability)	
		B.append(dataset[np.random.choice(n,1,p=probability)[0]])

	return B

def CoresetConstruction(dataset, cluster_number, D2sampleset, coreset_size, distance_func):
	n = len(dataset)
	alpha = 16*(math.log10(k)+2)
 
	dist_mat = distance.cdist(dataset, D2sampleset, distance_func) # n*k array, xij:distance between dataset[i] and D2sampleset[j]
	labels = np.argmin(dist_mat, axis=1)
	partition = [[] for i in range(cluster_number)]
	for i in range(n):
		partition[labels[i]].append(dataset[i])
	
	c_phi = np.sum(np.amin(dist_mat, axis=1))/n
	probability = np.zeros(n)
	for i in range(n):
		term1 = alpha*np.amin(distance.cdist([dataset[i]], D2sampleset, distance_func),axis=1)[0]/c_phi
		term2 = 2*alpha*np.sum(np.amin(distance.cdist(partition[labels[i]], D2sampleset),axis=1))/(len(partition[labels[i]])*c_phi)
		term3 = 4*n/len(partition[labels[i]])
		probability[i] = term1 + term2 + term2
	probability = probability/np.sum(probability)
	weight = 1.0/(coreset_size*probability)
	Cindex = np.random.choice(n, coreset_size, replace=False, p=probability)

	return Cindex, [dataset[i] for i in Cindex], [weight[i] for i in Cindex]


parser = argparse.ArgumentParser(description='CS coreset construction')
parser.add_argument('filename', type=str, help='file name')
parser.add_argument('attributes', type=int, help='number of attributes')
parser.add_argument('cluster_number', type=int, help='number of cluster')
parser.add_argument('coreset_size', type=int, help='coreset size')
parser.add_argument('-coreset_number', type=int, default=1, required=False, help='number of coresets to generate')
parser.add_argument('-seperator', type=str, default='\t', required=False, choices=['\t',' ',','], help='the character used to seperate values')
parser.add_argument('-start_row', type=int, default=0, required=False, help='the row at which the data start')
parser.add_argument('-start_col', type=int, required=False, help='the column at which the data start')
parser.add_argument('-end_col', type=int, required=False, help='the column at wich the data end')
parser.add_argument('-distance_func', type=str, default='mahalanobis', required=False, help='function to calculate the distance')

args = parser.parse_args()
filename = args.filename
p = args.attributes
k = args.cluster_number
m = args.coreset_size

for t in range(args.coreset_number):
	dataset = ReadInData(filename, p, args.seperator, args.start_row, args.start_col, args.end_col)
	B = D2Sampling(dataset, k, args.distance_func)
	index, coreset, weight = CoresetConstruction(dataset, k, B, m, args.distance_func)	

	with open(filename[:-4]+'_CS'+str(m)+'_'+str(t)+'.txt', 'w', encoding='utf-8') as outputfile:
		for i in range(m):
			for j in range(p):
				outputfile.write(str(coreset[i][j])+'\t')
			outputfile.write(str(weight[i])+'\n')

