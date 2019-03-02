'''
Created on Mar 1, 2019

@author: Wenxing Zhang, Ying Cai
'''
from sklearn.cluster import KMeans
from time import time
import argparse
import math

def kmeans(clusters, data, weight):
    t0 = time()
    KM = KMeans(n_clusters=clusters).fit(data,sample_weight=weight)
    usedtime=time()-t0
    return KM, usedtime

def eudistance(a, centroid):
    d=0
    for i in range(len(a)):
       d += math.pow((vector[i]-centroid[i]),2) 
    return d

def compute_quantization_error(data, labels, cluster_centers):
    error = 0
    for i in range(len(data)):
      error += eudistance(data[i], cluster_centers[labels[i]])
    return error

parser = argparse.ArgumentParser(description='Lightweight coreset construction')
parser.add_argument('numOfVariable', type=int, help='number of attributes')
parser.add_argument('filename', type=str, help='filename')
parser.add_argument('--seperator', type=str, default='\t', required=False, choices=['\t',' ',','], help='the character used to seperate values')
args = parser.parse_args()
numOfVariable = args.numOfVariable
filename = args.filename
seperator = args.seperator

dataset=[]
with open(filename) as fn:
    for line in fn:
        data = line.split(seperator)        
        data = data[1:]        
        vlist=[]
        for value in data:
            vlist.append(float(value))
        dataset.append(vlist)
   
fn.close()

clusters=[100,500]
for k in clusters:
    totalKM, totalT = kmeans(k, dataset, None)
    error = compute_quantization_error(dataset, totalKM.labels_, totalKM.cluster_centers_)
    with open(filename[:-4]+"_Full_k"+str(k)+".txt",'w',encoding='utf-8') as f:
        f.write("used time: "+str(totalT)+"\n# of Clusters:"+str(k)+"\nCluster\n")
        f.write('quantization error on the full data set: '+str(error)+'\n')
        f.write('\n\nlabels for full data set:\n')
        for item in totalKM.labels_:
            f.write(str(item)+'\n')
        f.write('\n\ncluster centroids:\n')
        for item in totalKM.cluster_centers_:
            for coord in item:
                f.write(str(item)+'\t')
            f.write('\n')

sampleSize=[1000,2000,5000]
repeat=[1,2,3]

for s in sampleSize:
    for r in repeat:
        subData=[]
        weights=[]
        with open(filename[:-4]+"_"+str(s)+"_"+str(r)+".txt") as f:
            for line in f:
                l=line.split('\t')
                index=int(l[0])
                weight=float(l[1].strip())
                subData.append(dataset[index])
                weights.append(weight)
            for k in clusters:
                predictKM, pTime = kmeans(k, subData, weights)
                labels = predictKM.predict(dataset)
                error = compute_quantization_error(dataset, labels, predictKM.cluster_centers_)
                with open(filename[:-4]+"_LWCS_"+str(s)+"_r_"+str(r)+"_k_"+str(k)+".txt",'w',encoding='utf-8') as f:
                    f.write("used time: "+str(pTime)+"\n# of Clusters:"+str(k)+"\nCluster\n")
                    f.write('quantization error on the full data set: '+str(error)+'\n')
                    f.write('\n\nlabels for full data set:\n')
                    for item in labels:
                        f.write(str(item)+'\n')    
                    f.write('\n\ncluster centroids:\n')
                    for item in predictKM.cluster_centers_:
                        for coord in item:
                           f.write(str(item)+'\t')
                        f.write('\n')
                    f.close()                    
                    
                    
                    
                    
                    
