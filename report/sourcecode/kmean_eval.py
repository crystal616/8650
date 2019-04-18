'''
Created on Mar 1, 2019
Perform Kmean++ on corsets
Results are evaluated by computing the quantization error on the full dataset

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
    parser.add_argument('corsetstype', type=str, choices=["LWCS", "CS", "UNIFORM"], help='corset construction method')
    parser.add_argument('numOfVariable', type=int, help='number of attributes')
    parser.add_argument('numOfCluster', type=int, help='number of clusters')
    parser.add_argument('samplesize', type=int, help='corset size')
    parser.add_argument('numOfCorsets', type=int, help='number of corsets')
    
    args = parser.parse_args()
    dataset = args.dataset
    corsetstype = args.corsetstype
    numOfVariable = args.numOfVariable
    numOfCorsets = args.numOfCorsets
    k = args.numOfCluster
    m = args.samplesize

    fulldata = ReadInData(dataset+".txt", numOfVariable)

    avg_error = 0
    avg_error2 = 0
    with open(dataset+"_"+corsetstype+"_"+str(m)+"_"+str(k)+"_Performance.txt", 'w') as f:
        for repeat in range(numOfCorsets):
            f.write("run "+str(repeat+1)+'\n')
            filename = dataset+"_"+corsetstype+"_"+str(m)+"_"+str(repeat+1)+".txt"
            if(corsetstype == "UNIFORM"):
                data = ReadInData(filename, numOfVariable)
                predictKM, pTime = kmeans(k, data, None)
            else:
                data, weights = ReadInDataWWeight(filename, numOfVariable)
                predictKM, pTime = kmeans(k, data, weights)
            f.write("used time: "+str(pTime)+'\n')
            labels = predictKM.predict(fulldata)
            error = compute_quantization_error(fulldata, labels, predictKM.cluster_centers_)
            f.write("quantization error on the full data set: "+str(error)+'\n')
            avg_error += error
            avg_error2 += error**2
        avg_error /= numOfCorsets
        avg_error2 /= numOfCorsets
        var_error = avg_error2 - avg_error**2
        f.write("\n\navg_error: "+str(avg_error)+'\n'+"var_error: "+str(var_error))
    f.close()
    

