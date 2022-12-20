#!/usr/bin/env python3

from os import sep
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
import scipy

def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    for i in range(n_clusters):
        idx = labels == i
        # Choose the most common label among data points in the cluster
        new_label=scipy.stats.mode(real_labels[idx])[0][0]
        permutation.append(new_label)
    return permutation


def nonconvex_clusters():
    pass

def main():
    print(nonconvex_clusters())

if __name__ == "__main__":
    main()
