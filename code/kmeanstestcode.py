'''
Created on Feb 27, 2019
Construct coresets by uniform sampling or by LWCS algorithm.
Analyse their performances by computing quantization errors.

(LWCS algorithm:
Sampling m weighted points from datasets. Each point x has weight 1/(m*q(x)) 
and is sampled with probability q(x)=1/2n+(d〖(x,μ)〗^2)/(2∑_(x^'∈X)▒〖d(x^',μ)〗^2 )  
where μ is the mean of the whole dataset X.

@author: Ying Cai, Wenxing Zhang
'''
from multiprocessing import Pool
from _functools import partial
from sklearn.cluster import KMeans
from time import time
import math
import numpy as np
import argparse

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
    for vector in a:
        for i in range(numOfVariable):
            d += math.pow((vector[i]-centroid[i]),2) 
    return d

def sumvectors(a,numofvar):
    r=[]

    for _i12 in range(numofvar):
        r.append(0)
    for vector in a:
        for k in range(numofvar):
            r[k] = r[k] + vector[k]  
    return r

def probability(a, centroid, n, totaldistance):
        
    for vector in a:
        d=0
        for i in range(numOfVariable):
            d += math.pow((vector[i]-centroid[i]),2) 
        vector.append(0.5/n + 0.5*d/totaldistance)
    return a
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Lightweight coreset construction')
    parser.add_argument('numOfVariable', type=int, help='number of attributes')
    parser.add_argument('excludeVariable',type=int,help='number of attributes which need to be excluded')
    parser.add_argument('numOfCore', type=int, help='number of cores to use')
    parser.add_argument('filename', type=str, help='file name')
    parser.add_argument('--seperator', type=str, default='\t', required=False, choices=['\t',' ',','], help='the character used to seperate values')
    
    args = parser.parse_args()
    numOfVariable = args.numOfVariable
    numOfCore = args.numOfCore
    filename = args.filename
    seperator = args.seperator
    excludeVariable = args.excludeVariable
   
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
    
    
    nsize = len(dataset)

    splitData = int(nsize/numOfCore)
    
    print("size n:", nsize)
    print('\n\n')   
 
    y=[]
    
    for numOfSubSet in range(numOfCore-1):
        y.append(dataset[numOfSubSet*splitData: (numOfSubSet+1)*splitData])
    
    y.append(dataset[(numOfSubSet+1)*splitData:])    
    
    with Pool(processes=numOfCore) as p:
        result=p.map(partial(sumvectors,numofvar=numOfVariable), y)
      
    sumv = sumvectors(result,numOfVariable)

    print("sum: ", sumv)
    print('\n\n')

    center = []
     
    for i13 in range(numOfVariable):
        center.append(0)
        center[i13] = sumv[i13]/nsize
     
    print("center: ", center)
    print('\n\n')
     
    with Pool(processes=numOfCore) as p:
        distances = p.map(partial(eudistance, centroid=center), y)
     
     
    totald = 0
    for value in distances:
        totald += value
     
    print("total distances:",totald)
    
    f=open(filename[:-4]+'_Statistics.txt','w')
    f.write("size n:" + str(nsize) + "\nsum: "+str(sumv)+"\ncenter: "+str(center)+"\ntotal distance: "+str(totald))
    f.close()
 
    with Pool(processes=numOfCore) as p:
        addp = p.map(partial(probability, centroid=center, n=nsize, totaldistance = totald), y)
  
     
    z=[]
    for l in addp:
        z.extend(l)
      
    posibilities=[]
    
    for vector in z:
        posibilities.append(vector[numOfVariable])
    
    size=[]
  
    for x in range(nsize):
        size.append(x)
       
    
    sampleSize = [1000,2000]
    repeat = 2
    clusters=[100,500]
    
    #lightweaight coreset
    for s in sampleSize:
		variances=[]		
        for sampleTimes in range(repeat):	
			varinces[sampleTimes]=[]
            chosen=np.random.choice(size, s, replace=False, p=posibilities)    
            subData=[]
            weights=[]
            with open(filename[:-4]+'_'+str(s)+"_"+str(sampleTimes+1) + " LWCS.txt",'w',encoding='utf-8') as f:
                for item in chosen:
                    f.write(str(item)+"\t"+str(1/(s*float(posibilities[item])))+'\n')
                    subData.append(dataset[item])
                    weights.append(1/(s*float(posibilities[item])))
            f.close()
            for k in clusters:
                predictKM, pTime = kmeans(k, subData, weights)
                labels = predictKM.predict(dataset)
                error = compute_quantization_error(dataset, labels, predictKM.cluster_centers_)
				varinces[sampleTimes].append(error)
                with open(filename[:-4]+"_LWCS_"+str(s)+"_r_"+str(sampleTimes+1)+"_k_"+str(k)+".txt",'w',encoding='utf-8') as f:
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
        with open(filename[:-4]+'_'+str(s)+"_" + "Variances LWCS.txt",'w',encoding='utf-8') as f:
			f.write("Sample Size: "+str(s)+"\n")
			f.write("Clusters=100\tClusters=500\n")
            for item in variances:
				for v in item:
					f.write(str(v)+"\t")
				f.write("\n")
        f.close()  
    
    #uniform sampling
    for s in sampleSize:
		variances=[]
        for sampleTimes in range(repeat):
			varinces[sampleTimes]=[]
            chosen = np.random.choice(size, s, replace=False)
            subData=[]
            with open(filename[:-4]+'_'+str(s)+"_"+str(sampleTimes+1) + " Uniform.txt",'w',encoding='utf-8') as f:
                for item in chosen:
                    f.write(str(item)+'\n')
                    subData.append(dataset[item])
            f.close()
            for k in clusters:
                predictKM, pTime = kmeans(k, subData, None)
                labels = predictKM.predict(dataset)
                error = compute_quantization_error(dataset, labels, predictKM.cluster_centers_)
				varinces[sampleTimes].append(error)
                with open(filename[:-4]+"_Uniform_"+str(s)+"_r_"+str(sampleTimes+1)+"_k_"+str(k)+".txt",'w',encoding='utf-8') as f:
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
		with open(filename[:-4]+'_'+str(s)+"_" + "Variances Uniform.txt",'w',encoding='utf-8') as f:
			f.write("Sample Size: "+str(s)+"\n")
			f.write("Clusters=100\tClusters=500\n")
            for item in variances:
				for v in item:
					f.write(str(v)+"\t")
				f.write("\n")
        f.close()
		
		
    #CS coreset
    
    
