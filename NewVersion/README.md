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
2) CS
3) UNIFROM

50 corsets of size 1000, 2000, 5000 are generated using each of those three methods. For CS, number of clusters k = 100, 500, were used.

## Performance evaluation
(kmean\_eval.py, kmedoid\_eval.py, kmean\_full.py, kmedoid\_full.py)

Both Kmeans and Kmedoids were ran on those corests. The quantization error on the full dataset were computed and compared to the results from Kmeans and Kmedoids ran on the full dataset.
