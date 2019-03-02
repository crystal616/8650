'''
Created on Mar 1, 2019

@author: Ying
'''
from sklearn.cluster import KMeans
from time import time


numOfVariable=74

def kmeans(clusters, data, weight):
    t0 = time()
    KM = KMeans(n_clusters=clusters).fit(data, sample_weight=weight)
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

filename="Bio_train\\bio_test.dat"
testset=[]
with open(filename) as fn:
    for line in fn:
        data = line.split('\t')        
        data = data[3:]        
        data[numOfVariable-1] = data[numOfVariable-1].strip()
        vlist=[]
        for value in data:
            vlist.append(float(value))
        testset.append(vlist)
   
fn.close()

clusters=[100,500]
print("done")
for k in clusters:
    totalKM, totalT = kmeans(k, dataset, None)
    labels = totalKM.predict(testset)
    with open("Bio_train\\Full set k="+str(k)+".txt",'w',encoding='utf-8') as f:
        f.write("used time: "+str(totalT)+"\n# of Clusters:"+str(k)+"\nCluster\n")
        for item in labels:
            f.write(str(item)+'\n')    
        f.close()

sampleSize=[1000,2000,5000]
repeat=[1,2,3]

for s in sampleSize:
    for r in repeat:
        subData=[]
        weights=[]
        with open("Bio_train\\Bio_train "+str(s)+" "+str(r)+".txt") as f:
            for line in f:
                l=line.split('\t')
                index=int(l[0])
                weight=float(l[1].strip())
                subData.append(dataset[index])
                weights.append(weight)
            print("done")
            for k in clusters:
                predictKM, pTime = kmeans(k, subData, weights)
                predict_lable = predictKM.predict(testset)
                with open("Bio_train\\LWCS "+str(s)+" r="+str(r)+" k="+str(k)+".txt",'w',encoding='utf-8') as f:
                    f.write("used time: "+str(pTime)+"\n# of Clusters:"+str(k)+"\nCluster\n")
                    for item in predict_lable:
                        f.write(str(item)+'\n')    
                    f.close()                    
                    
                    
                    
                    
                    
