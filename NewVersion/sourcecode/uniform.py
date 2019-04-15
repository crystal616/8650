'''
Created on Feb 27, 2019
construct corest using "uniform" subsampling method

@author: Ying Cai, Wenxing Zhang
'''
import numpy as np
import argparse

def fwrite_vector(f, vec):
    for item in vec:
        f.write(str(item)+' ')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Lightweight coreset construction')
    parser.add_argument('filename', type=str, help='file name')
    parser.add_argument('numOfVariable', type=int, help='number of attributes')   
    parser.add_argument('samplesize', type=int, help='coreset size')
    parser.add_argument('--numOfCorset', type=int, default=1, help='number of coresets to construct')
    args = parser.parse_args()
    numOfVariable = args.numOfVariable
    filename = args.filename
    samplesize = args.samplesize
    numOfCorset = args.numOfCorset
   
    dataset=[]
    with open(filename) as fn:
        for line in fn:
            data = line.split()
            data[numOfVariable-1] = data[numOfVariable-1].strip()
            vlist=[]
            for value in data:
                vlist.append(float(value))
            dataset.append(vlist)

    fn.close()
    
    
    nsize = len(dataset)
    
    size=[]
  
    for x in range(nsize):
        size.append(x)
       
    for sampleTimes in range(numOfCorset):
        chosen = np.random.choice(size, samplesize, replace=False)
        with open(filename[:-4]+'_UNIFORM_'+str(samplesize)+"_"+str(sampleTimes+1) + ".txt",'w',encoding='utf-8') as f:
            for item in chosen:
                fwrite_vector(f, dataset[item])
                f.write('\n')
        f.close()
