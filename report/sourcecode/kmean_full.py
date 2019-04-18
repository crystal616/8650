'''
Created on Mar 1, 2019

@author: Wenxing Zhang, Ying Cai
'''

from sklearn.cluster import KMeans
from time import time
import math
import numpy as np
import argparse
import sys

def ReadInData(filename, numOfVariable):
    with open(filename) as datafile:
        dataset = []
        for line in datafile:
            data = line.split()        
            if(len(data) != numOfVariable):
                print("The number of attributes specified does not match the data! Exit the program!")
                sys.exit()
            vlist=[]
            for value in data:
                vlist.append(float(value.strip()))
            dataset.append(vlist)
    datafile.close()
    return dataset

def ReadInDataWWeight(filename, numOfVariable):
    with open(filename) as datafile:
        dataset = []
        weights = []
        for line in datafile:
            data = line.split()        
            if(len(data) != numOfVariable+1):
                print("The number of attributes specified does not match the data! Exit the program!")
                sys.exit()
            weights.append(float(data[0].strip()))
            vlist=[]
            for value in data[1:]:
                vlist.append(float(value.strip()))
            dataset.append(vlist)
    datafile.close()
    return dataset, weights

def kmeans(clusters, data, weight):
    t0 = time()
    KM = KMeans(n_clusters=clusters).fit(data,sample_weight=weight)
    usedtime=time()-t0
    return KM, usedtime

def compute_quantization_error(data, labels, cluster_centers):
    error = 0
    for i in range(len(data)):
        error += eudistance(data[i], cluster_centers[labels[i]])
    return error

def eudistance(a, centroid):
    d=0
    for i in range(len(a)):
        d += (a[i]-centroid[i])**2 
    return d

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='performance of kmeans via corsets')
    parser.add_argument('dataset', type=str, choices=["kdd","song"], help='dataset name')
    parser.add_argument('numOfVariable', type=int, help='number of attributes')
    parser.add_argument('numOfCluster', type=int, help='number of clusters')
    
    args = parser.parse_args()
    dataset = args.dataset
    numOfVariable = args.numOfVariable
    k = args.numOfCluster

    fulldata = ReadInData(dataset+".txt", numOfVariable)
    predictKM, pTime = kmeans(k, fulldata, None)
    labels = predictKM.predict(fulldata)
    error = compute_quantization_error(fulldata, labels, predictKM.cluster_centers_)
    with open(dataset+"_kmean_"+str(k)+"_Performance.txt", 'w') as f:
        f.write("used time: "+str(pTime)+'\n')
        f.write("quantization error on the full data set: "+str(error)+'\n')
    f.close()
    

