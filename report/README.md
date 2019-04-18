# 8650 Final Project
Light-weight Coreset Generation and Performance Analysis

Reproduce the results of paper "Scalable k-Means Clustering via Lightweight Coresets", Olivier Bachem, Mario Lucic, Andreas Krause. KDD 2018, August 19-23, 2018, London, United Kingdom.

## Dataset

Two datasets are analyzed in this study.
1) KDD/bio\_train.data — 145’751 samples with 74 features measuring the match between a protein and a native sequence (http://osmot.cs.cornell.edu/kddcup/datasets.html)
2) SONG/YearPredictionMSD.txt — 90 features from 515’345 songs of the Million Song datasets used for predicting the year of songs (http://archive.ics.uci.edu/ml/datasets/yearpredictionmsd)

## Data preprocessing

Using "awk" to convert the original dataset file to a file containing only feature values seperated by whitespace.

KDD/bio\_train.data -> KDD/kdd.txt

SONG/YearPredictionMSD.txt -> SONG/song.txt

(files are not uploaded due to size limitation)

## Corset constructions
(lightweightCorset.py, strongCorset.py, uniform.py)

Corsets are generated using three methods:
1) LWCS 

usage: `ligtweightCoresets.py [-h] [--numOfCorset NUMOFCORSET]
                             [--numOfCore NUMOFCORE]
                             numOfVariable filename samplesize`

Lightweight coreset construction

positional arguments:
*  `numOfVariable`         number of attributes
*  `filename`              file name
*  `samplesize`            coreset size

optional arguments:
*  `-h, --help`            show this help message and exit
*  `--numOfCorset NUMOFCORSET`
                        number of coresets to construct
*  `--numOfCore NUMOFCORE`
                        number of cores to use

2) CS

usage: `strongCorsets.py [-h] [--numOfCorset NUMOFCORSET]
                        [--distance_func DISTANCE_FUNC]
                        filename numOfVariable cluster_number samplesize`

CS coreset construction

positional arguments:
 * `filename`              file name
 * `numOfVariable`         number of attributes
 * `cluster_number`        number of cluster
 * `samplesize`            coreset size

optional arguments:
 * `-h, --help`            show this help message and exit
 * `--numOfCorset NUMOFCORSET`
                        number of coresets to construct
 * `--distance_func DISTANCE_FUNC`
                        function to calculate the distance

3) UNIFROM

usage: `uniform.py [-h] [--numOfCorset NUMOFCORSET]
                  filename numOfVariable samplesize`

Uniformly subsampling

positional arguments:
 * `filename`              file name
 * `numOfVariable`         number of attributes
 * `samplesize`            coreset size

optional arguments:
 * `-h, --help`            show this help message and exit
 * `--numOfCorset NUMOFCORSET`
                        number of coresets to construct
                       
50 corsets of size 1000, 2000, 5000 are generated using each of those three methods. For CS, number of clusters k = 100, 500, were used.

## Performance evaluation
(kmean\_eval.py, kmedoid\_eval.py, kmean\_full.py, kmedoid\_full.py)

usage: `kmean_eval.py [-h]
                     {kdd,song} {LWCS,CS,UNIFORM} numOfVariable numOfCluster
                     samplesize numOfCorsets`

performance of kmeans via corsets

positional arguments:
 * `{kdd,song}`         dataset name
 * `{LWCS,CS,UNIFORM}`  corset construction method
 * `numOfVariable`      number of attributes
 * `numOfCluster`       number of clusters
 * `samplesize`         corset size
 * `numOfCorsets`       number of corsets

optional arguments:
 * `-h, --help`         show this help message and exit

usage: `kmedoid_eval.py [-h] [--maxiterations MAXITERATIONS]
                       {kdd,song} {LWCS,CS,UNIFORM} numOfVariable numOfCluster
                       samplesize numOfCorsets`

performance of kmeans via corsets

positional arguments:
 * `{kdd,song}`            dataset name
 * `{LWCS,CS,UNIFORM}`     corset construction method
 * `numOfVariable`         number of attributes
 * `numOfCluster`          number of clusters
 * `samplesize`            corset size
 * `numOfCorsets`          number of corsets

optional arguments:
 * `-h, --help`            show this help message and exit
 * `--maxiterations MAXITERATIONS`
                        max number of iterations

Both Kmeans and Kmedoids were ran on those corests. The quantization error on the full dataset were computed and compared to the results from Kmeans and Kmedoids ran on the full dataset.
