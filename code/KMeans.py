'''
Created on Mar 1, 2019

@author: Ying
'''
from sklearn.cluster import KMeans
# import random
from time import time


numOfVariable=74

# def generate(n):
#     g = []
#     for _i in range(n):
#         g.append(random.randint(1, 100))
#     return g

def kmeans(clusters, data):
    t0 = time()
    KM = KMeans(n_clusters=clusters).fit(data)
    usedtime=time()-t0
    return KM, usedtime

filename="Bio_train\\bio_train.dat"
dataset=[]
with open(filename) as fn:
    for line in fn:
        data = line.split('\t')        
        data = data[3:]        
        data[numOfVariable-1] = data[numOfVariable-1].strip()
        vlist=[]
        for value in data:
            vlist.append(float(value))
        dataset.append(vlist)
  
fn.close()
# dataset=[]
# for _i in range(1000):
#     dataset.append(generate(10))

clusters=[100,500]

for k in clusters:
    totalKM, totalT = kmeans(k, dataset)
    labels = totalKM.labels_
    with open("Bio_train\\Total set k="+str(k)+".txt",'w',encoding='utf-8') as f:
        f.write("used time: "+str(totalT)+"\n# of Clusters:"+str(k)+"\nCluster\n")
        for item in labels:
            f.write(str(item)+'\n')    
        f.close()
    

