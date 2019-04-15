'''
Created on Mar 30, 2019

Perform Kmedoid on corsets
Results are evaluated by computing the quantization error on the full dataset

@author: Ying Cai, Wenxing Zhang
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

def kmedoids(k, data, weight, maxiteration):      
    t0 = time()
    currentcenter = np.random.choice(len(data), k, replace=False)
    #print("initialcenter: "+str(currentcenter)+"\n")
    usediterations=0 
    for i in range(maxiteration):
        temp=currentcenter
        currentcluster=assignclusters(currentcenter, data)
        currentcenter, centerchanged=optimalcenter(data, temp, currentcluster, weight, k)
        usediterations = i
        if not centerchanged:
            break
        
    currentcluster=assignclusters(currentcenter, data)
    usedtime=time()-t0
    return currentcenter, currentcluster, usedtime, usediterations

def assignclusters(currentcenter,data):
    currentcluster=[]    
    for p in range(len(data)):              
        mindistance=sys.maxsize
        assignto=-1
        for i in range(len(currentcenter)):
            distance = point_eudistance(data[p],data[currentcenter[i]])
            if distance < mindistance:
                mindistance = distance
                assignto = i
        currentcluster.append(assignto)
    #print("currentcluster: "+str(currentcluster)+"\n")
    return currentcluster

def assignfullsetclusters(fullset, subset, cluster_centers):
    clusters=[]    
    for p in range(len(fullset)):             
        mindistance=sys.maxsize
        assignto=-1
        for i in range(len(cluster_centers)):
            distance = point_eudistance(fullset[p],subset[cluster_centers[i]])
            if distance < mindistance:
                mindistance = distance
                assignto = i
        clusters.append(assignto)
    #print("full set cluster: "+str(clusters)+"\n")
    return clusters

def optimalcenter(data, currentcenters, currentcluster, weight, k):
    optimalcenter=[]
    members=[[] for _i in range(k)]   
    for p in range(len(data)):
        members[currentcluster[p]].append(p)
    
    centerchanged = False
    for p in range(len(currentcenters)):        
        member=members[p]
        center=currentcenters[p]
        new_center=center
        currentcost=0
        for m in member:
            currentcost += weight[m]*point_eudistance(data[m],data[center])
        ##print("currentcost:" + str(currentcost))
        for m in member:            
            newcost=0
            for other in member:
                newcost += weight[other]*point_eudistance(data[other],data[m])     
            ##print("newcost: "+str(newcost))           
            if newcost < currentcost:
                centerchanged = True
                currentcost = newcost
                ##print("before, center: "+str(new_center))
                new_center = m
                ##print("after, center: "+str(new_center)+"\n")
        ##print("optimal center: "+str(new_center)+"\n")
        optimalcenter.append(new_center)
    return optimalcenter, centerchanged

def point_eudistance(a, centroid):
    d=0
    for i in range(numOfVariable):
        d += (a[i]-centroid[i])**2
    return d

def compute_quantization_error(data, labels, subset, cluster_centers):
    error = 0
    for i in range(len(data)):
        error += point_eudistance(data[i], subset[cluster_centers[labels[i]]])
    return error

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='performance of kmeans via corsets')
    parser.add_argument('dataset', type=str, choices=["kdd","song"], help='dataset name')
    parser.add_argument('corsetstype', type=str, choices=["LWCS", "CS", "UNIFORM"], help='corset construction method')
    parser.add_argument('numOfVariable', type=int, help='number of attributes')
    parser.add_argument('numOfCluster', type=int, help='number of clusters')
    parser.add_argument('samplesize', type=int, help='corset size')
    parser.add_argument('numOfCorsets', type=int, help='number of corsets')
    parser.add_argument('--maxiterations',type=int, default=300, help='max number of iterations')
    
    args = parser.parse_args()
    dataset = args.dataset
    corsetstype = args.corsetstype
    numOfVariable = args.numOfVariable
    numOfCorsets = args.numOfCorsets
    maxiterations = args.maxiterations
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
                weights = np.ones(len(data))
            else:
                data, weights = ReadInDataWWeight(filename, numOfVariable)
            centers, clusters, pTime, usediterations = kmedoids(k, data, weights, maxiterations)
            f.write("used time: "+str(pTime)+'\n')
            labels = assignfullsetclusters(fulldata, data, centers)
            error = compute_quantization_error(fulldata, labels, data, centers)
            f.write("quantization error on the full data set: "+str(error)+'\n')
            avg_error += error
            avg_error2 += error**2
        avg_error /= numOfCorsets
        avg_error2 /= numOfCorsets
        var_error = avg_error2 - avg_error**2
        f.write("\n\navg_error: "+str(avg_error)+'\n'+"var_error: "+str(var_error))
    f.close()
    

