'''
Created on Feb 27, 2019

@author: Ying Cai, Wenxing Zhang
'''
from multiprocessing import Pool
from _functools import partial
import math
import numpy as np

import argparse

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
    parser.add_argument('numOfCore', type=int, help='number of cores to use')
    parser.add_argument('filename', type=str, help='file name')
    parser.add_argument('--seperator', type=str, default='\t', required=False, choices=['\t',' ',','], help='the character used to seperate values')
    args = parser.parse_args()
    numOfVariable = args.numOfVariable
    numOfCore = args.numOfCore
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
      
    sum = sumvectors(result,numOfVariable)

    print("sum: ", sum)
    print('\n\n')

    center = []
     
    for i13 in range(numOfVariable):
        center.append(0)
        center[i13] = sum[i13]/nsize
     
    print("center: ", center)
    print('\n\n')
     
    with Pool(processes=numOfCore) as p:
        distances = p.map(partial(eudistance, centroid=center), y)
     
     
    totald = 0
    for value in distances:
        totald += value
     
    print("total distances:",totald)
    
    f=open(filename[:-4]+'_Statistics.txt','w')
    f.write("size n:" + str(nsize) + "\nsum: "+str(sum)+"\ncenter: "+str(center)+"\ntotal distance: "+str(totald))
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
       
    
    sampleSize = [1000,2000,5000]
    for s in sampleSize:
        for sampleTimes in range(3):
            chosen=np.random.choice(size, s, replace=False, p=posibilities)
            with open(filename[:-4]+'_'+str(s)+"_"+str(sampleTimes+1) + ".txt",'w',encoding='utf-8') as f:
                for item in chosen:
                    f.write(str(item)+"\t"+str(1/(s*float(posibilities[item])))+'\n')
            f.close()
     

