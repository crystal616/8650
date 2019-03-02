'''
Created on Mar 1, 2019

@author: Ying
'''

from sklearn import metrics

numOfVariable=74
 
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
 
clusters=[100,500]
sampleSize=[1000,2000,5000]
repeat=[1,2,3]
 
for k in clusters:
    true_lable=[]
    with open("Bio_train\\clusters\\Full set k="+str(k)+".txt") as f:
        next(f)
        next(f)
        next(f)
        for line in f:
            value=line.strip()
            true_lable.append(int(value))
    f.close()
     
    for s in sampleSize:
        for r in repeat:
            pred_lable=[]
            with open("Bio_train\\clusters\\LWCS "+str(s)+" r="+str(r)+" k="+str(k)+".txt") as f:
                next(f)
                next(f)
                next(f)
                for line in f:
                    value=line.strip()
                    pred_lable.append(int(value))
            f.close()
     
            with open("Bio_train\\metrics\\LWCS "+str(s)+" r="+str(r)+" k="+str(k)+".txt",'w',encoding='utf-8') as f:
                f.write("homogeneity_score: "+str(metrics.homogeneity_score(true_lable, pred_lable))
                        +'\ncompleteness_score: '+str(metrics.completeness_score(true_lable, pred_lable))
                        +"\nv_measure_score: "+str(metrics.v_measure_score(true_lable, pred_lable))
                        +"\nadjusted_rand_score: "+str(metrics.adjusted_rand_score(true_lable, pred_lable))
                        +"\nadjusted_mutual_info_score: "+str(metrics.adjusted_mutual_info_score(true_lable, pred_lable))
                        +"\nfowlkes_mallows_score:"+str(metrics.fowlkes_mallows_score(true_lable, pred_lable)))
            f.close()

# true_label=[0,0,1,1,2,2]
# pred_lable=[0,1,2,2,2,1]
# 
# print("homogeneity_score: "+str(metrics.homogeneity_score(true_label, pred_lable))
#     +'\ncompleteness_score: '+str(metrics.completeness_score(true_label, pred_lable))
#     +"\nv_measure_score: "+str(metrics.v_measure_score(true_label, pred_lable))
#     +"\nadjusted_rand_score: "+str(metrics.adjusted_rand_score(true_label, pred_lable))
#     +"\nadjusted_mutual_info_score: "+str(metrics.adjusted_mutual_info_score(true_label, pred_lable))
#     +"\nfowlkes_mallows_score:"+str(metrics.fowlkes_mallows_score(true_label, pred_lable)))
                            
