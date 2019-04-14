'''
Created on Mar 30, 2019
Construct coresets by uniform sampling or by LWCS algorithm.
Analyse their performances by computing quantization errors.

@author: Ying Cai, Wenxing Zhang
'''

from time import time
import math
import numpy as np
import argparse
import sys
       
def kmedoids(k, data, maxiteration):      
    t0 = time()
    currentcenter = np.random.choice(len(data), k, replace=False)   
    usediterations=0 
    for _i in range(maxiteration):
        currentcluster=assignclusters(currentcenter, data)
        currentcenter, tolerance=optimalcenter(data, currentcenter, currentcluster, k)
        usediterations += 1
        if tolerance:
            break
    currentcluster=assignclusters(currentcenter, data)
    usedtime=time()-t0
    return currentcenter, currentcluster, usedtime

def assignclusters(currentcenter,data):
    currentcluster=[]    
    for p in range(len(data)):        
        mindistance=sys.maxsize
        assignto=-1
        for i in range(len(currentcenter)):
            distance=point_eudistance(data[p],data[currentcenter[i]])
            if distance<mindistance:
                mindistance=distance
                assignto=i
        currentcluster.append(assignto)
    return currentcluster

def optimalcenter(data, currentcenters, currentcluster, k):
    tolerance=True
    optimalcenter=[]
    members=[[] for _i in range(k)]
    ##for p in range(len(currentcenters)):
    ##    members[p] = []   
    for p in range(len(data)):
        members[currentcluster[p]].append(p)
    
    for p in range(len(currentcenters)):
        member=members[p]
        center=currentcenters[p]
        currentcost=0
        for m in member:
            currentcost += point_eudistance(data[m],data[center])
        for m in member:
            newcost=0
            for other in member:
                newcost += point_eudistance(data[other],data[m])
            if newcost<currentcost and abs(newcost-currentcost)<0.01 :
                tolerance=False
                currentcost=newcost
                center=m
        optimalcenter.append(center)
    return optimalcenter, tolerance


   

def compute_quantization_error(data, labels, cluster_centers):
    error = 0
    for i in range(len(data)):
        error += point_eudistance(data[i], data[cluster_centers[labels[i]]])
    return error

def point_eudistance(a, centroid):
    d=0
    for i in range(numOfVariable):
        d += math.pow((a[i]-centroid[i]),2) 
    return d  
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Lightweight coreset construction')
    parser.add_argument('numOfVariable', type=int, help='number of attributes')
    parser.add_argument('excludeVariable',type=int,help='number of attributes which need to be excluded')
    parser.add_argument('numOfCore', type=int, help='number of cores to use')
    parser.add_argument('filename', type=str, help='file name')
    parser.add_argument('maxiterations',type=int,help='max number of iterations')
    parser.add_argument('--seperator', type=str, default='\t', required=False, choices=['\t',' ',','], help='the character used to seperate values')
    
    args = parser.parse_args()
    numOfVariable = args.numOfVariable
    numOfCore = args.numOfCore
    filename = args.filename
    seperator = args.seperator
    excludeVariable = args.excludeVariable
    maxiterations=args.maxiterations
   
    dataset=[]
    with open(filename) as fn:
        for line in fn:
            data = line.split(seperator)            
            data = data[excludeVariable:]  
            data[numOfVariable-1] = data[numOfVariable-1].strip()          
            vlist=[]
            for value in data:
                vlist.append(float(value))
            dataset.append(vlist)

    fn.close()
    
    centers, clusters, pTime = kmedoids(100, dataset, maxiterations)
    labels = assignclusters(centers, dataset)
    error = compute_quantization_error(dataset, labels, centers)
    
    with open(filename[:-4]+"_kmedoids_fullset_100.txt",'w',encoding='utf-8') as f:
        f.write("used time: "+str(pTime)+"\n# of Clusters:100\nCluster\n")
        f.write('quantization error on the full data set: '+str(error)+'\n')
        f.write('\n\nlabels for full data set:\n')
        for item in labels:
            f.write(str(item)+'\n')    
        f.write('\n\ncluster centroids:\n')
        for item in centers:
            f.write(str(item)+'\n')                        
        f.close()
    
    centers, clusters, pTime = kmedoids(500, dataset, maxiterations)
    labels = assignclusters(centers, dataset)
    error = compute_quantization_error(dataset, labels, centers)
    
    with open(filename[:-4]+"_kmedoids_fullset_500.txt",'w',encoding='utf-8') as f:
        f.write("used time: "+str(pTime)+"\n# of Clusters:500\nCluster\n")
        f.write('quantization error on the full data set: '+str(error)+'\n')
        f.write('\n\nlabels for full data set:\n')
        for item in labels:
            f.write(str(item)+'\n')    
        f.write('\n\ncluster centroids:\n')
        for item in centers:
            f.write(str(item)+'\n')                        
        f.close()
    
    
