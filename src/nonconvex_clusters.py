#!/usr/bin/env python3

from os import sep
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
import scipy

def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    
    #Loop through values 
    for i in range(n_clusters):
        
        idx = labels == i
        # Choose the most common label among data points in the cluster
        new_label=scipy.stats.mode(real_labels[idx])[0][0]
        permutation.append(new_label)
    #After looping, we have a list where indexes are model labels and values are original labels
    #Think of this like a very basic dictionary
    return permutation


def nonconvex_clusters():
    data = pd.read_csv("src/data.tsv", sep = "\t")
    X = data[["X1","X2"]]
    y = data["y"]
    
    #Create a range of values between 0.05 and .2 at increments of 0.05
    #This will be eps value
    values = np.arange(0.05, 0.2, 0.05)
    results = []
    
    for value in values:
        #Creating a model based on eps value
        model = DBSCAN(value)
        model.fit(X)
        
        #Creating mask to exclude outlier data
        outlier_mask = model.labels_ != -1
        #Using mask to exclude outliers from original labels and from generated labels
        masked_labels = model.labels_[outlier_mask]
        masked_actual_labels = y[outlier_mask]
        #Find the number of outliers
        count_outliers = list(model.labels_).count(-1)
        
        #Number of clusters = number of unique labels, use set to find unique values
        number_clusters = len(set(masked_labels))
        
        #Create a list where index is generated label (with masking) and value is original label (with masking)
        permutation = find_permutation(number_clusters, masked_actual_labels, masked_labels)
        
        #Labels corrected using permutation list
        corrected_labels = [permutation[label] for label in masked_labels]
        
        #Accuracy scores
        acc = accuracy_score(masked_actual_labels, corrected_labels)
        
        
        #If number of clusters doesn't match original, then score is invalid
        if(number_clusters != len(set(y))):
            score = None
        else:
            score = acc
        
        #Add to result list
        results.append([value, score, number_clusters, count_outliers])
    
    #Transform result list into DataFrame
    answer = pd.DataFrame(results, columns = ["eps", "Score", "Clusters", "Outliers"], dtype = float)
    return answer
        

def main():
    print(nonconvex_clusters())

if __name__ == "__main__":
    main()
