import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import statistics
import copy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd


def BFP(data, time, dt):
    Vin_1 = data
    Vin_prev = 0
    Vout = 0
    Vout_1 = []
    RC = 5.3E-4
    for i in range(0, len(time) - 1):
        dVin = Vin_1[i] - Vin_prev
        dVout = dVin - dt * Vout / RC
        Vout = Vout + dVout
        Vout_1.append(Vout)
        Vin_prev = Vin_1[i]
    Vin_2 = Vout_1
    Vout = 0
    dVout = 0
    Vout_2 = [0]
    RC = 5.3E-5
    for i in range(0, len(time) - 1):
        dVout = dt * ((Vin_2[i] - Vout) / RC)
        Vout = Vout + dVout
        Vout_2.append(Vout)
    return Vout_2


def NEO_function(data):
    NEO = []
    for i in range(0, len(data)):
        if i == 0:
            neo = data[i] ** 2 - data[i + 1]
        elif i == len(data) - 1:
            neo = data[i] ** 2 - data[i - 1]
        else:
            neo = data[i] ** 2 - (data[i + 1] * data[i - 1])
        NEO.append(neo)
    return NEO


def Align_peaks(dt, fs, NEO_filtered, Train_filtered):

    pts_before = round(0.001 * fs)
    pts_after = round(0.002 * fs)
    t = np.arange(0, 0.003, dt)
    refractory = 2E-3
    refrac_pts = int(refractory * fs)
    APs = []
    i = 0
    fig = plt.figure()
    while i <= len(NEO_filtered) - refrac_pts:
        peak = []
        if NEO_filtered[i] >= threshold:
            window = [i, i + refrac_pts]
            peak = Train_filtered[window[0]:window[1]]
            max_value = max(peak)
            max_index = peak.index(max_value)
            max_index = max_index + i - 1
            window2 = [max_index - pts_before, max_index + pts_after]
            filt_window = Train_filtered[window2[0]:window2[1]]
            plt.plot(t, filt_window)
            APs.append(filt_window)
            i = i + refrac_pts
        else:
            i = i + 1

    plt.title('Aligned Action Potentials')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    fig.savefig('Aligned peaks.png')

    return(APs)


def PCA_analysis(AP):
    '''
    Extracting features using PCA analysis
    '''
    # setting PCA features as 2
    pca = PCA(n_components=2)
    # transforming the data
    transformed_data = pca.fit_transform(AP)

    return(transformed_data)


def cluster_Kmeans(data):
    '''
    Clustering based on KMeans
    '''
    # using KMeans function to identify clusters
    CLUSTER = KMeans(n_clusters=3).fit(data)

    # centroids = CLUSTER.cluster_centers_

    # assigning same integer values to data of same cluster
    color_indices = CLUSTER.predict(data)

    return(color_indices)


def cluster_assign(train_PCA, test_PCA, test_time, color_indices, fs):
    '''
    Clusters spikes in test data to any of the three neurons 
    3 clusters - 3 neurons
    Based on k nearest neighbours algorithm
    '''
    # considers 10 nearest neighbours
    k = 11
    dur = 0
    refractory_pd = round(0.002*fs)
    # storing size of rows for test and train data
    r_test = len(test_PCA)
    r_train = len(train_PCA)

    # intializing array to store data for the three neurons
    neuron_1 = np.zeros(len(test_time), dtype=int)
    neuron_2 = np.zeros(len(test_time), dtype=int)
    neuron_3 = np.zeros(len(test_time), dtype=int)

    # loop to identify and assign unklnown spikes
    # to any of the neurons

    for i in range(0, r_test):

        # to store indices of nearest neighbours
        cluster_check = []
        # to store distance between points
        d = []

        # calculating distances between points in extracted
        # features of training and test data
        for j in range(0, r_train):
            temp = LA.norm(train_PCA[j, :]-test_PCA[i, :])
            d.append(temp)

        # storing indices of k nearest neighbours
        idx = np.argpartition(d, k)[:k]

        # identifying which cluster it belongs to
        for j in range(0, k):
            cluster_check.append(color_indices[idx[j]])

        # looks for cluster with max occurence
        # and assigns to specific neuron
        # 0 - neuron_1, 1 - neuron_2 and 2 - neuron_3
        try :
            check = statistics.mode(cluster_check)

        except statistics.StatisticsError:
            # error occurs when there is 2 mode values
            # code to calculate mode
            c_0 = 0
            c_1 = 0
            c_2 = 0

            # calculating count of 0,1,2 in the cluster_check array
            for j in cluster_check:
                if cluster_check[j] == 0:
                    c_0 = c_0 + 1
                elif cluster_check[j] == 1:
                    c_1 = c_1 + 1
                elif cluster_check[j] == 2:
                    c_2 = c_2 + 1
                else:
                    continue

            c = [c_0, c_1, c_2]
            # getting index of the highest number in c
            check = np.argpartition(c,-1)[-1:]

        if check == 0:
            neuron_1[dur] = 1
            dur = dur + refractory_pd

        elif check == 1:
            neuron_2[dur] = 1
            dur = dur + refractory_pd

        elif check == 2:
            neuron_3[dur] = 1
            dur = dur + refractory_pd

        else:
            continue

    return (neuron_1, neuron_2, neuron_3)


if __name__ == "__main__":

    file = open('NII_Data.csv')
    line = next(file).strip('\n').split(',')
    fs = 2.23221E4
    dt = 1 / fs
    train_time = 5
    train_data = []
    test_data = []
    for i in range(0, int(train_time * fs)):
        train_data.append(float(line[i]))
    for i in range(int(train_time * fs) + 1, len(line)):
        test_data.append(float(line[i]))
    Train_t = np.arange(0.0, 5.0 - dt, dt)
    Test_t = np.arange(5.0, 25.0, dt)

    Train_filtered = BFP(train_data, Train_t, dt)

    fig = plt.figure()
    plt.plot(Train_t, Train_filtered)
    plt.title('Filtered Training Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    fig.savefig('Filtered Training Data.png')

    NEO_filtered = NEO_function(Train_filtered)

    fig = plt.figure()
    plt.plot(Train_t, NEO_filtered)
    plt.title('NEO Training Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Nonlinear Energy Operator')
    fig.savefig('NEO Training Data.png')

    var = statistics.median(np.abs(NEO_filtered)) / 0.67
    k = 4
    threshold = var * k

    AP = Align_peaks(dt, fs, NEO_filtered, Train_filtered)

    train_extracted_features = PCA_analysis(AP)

    fig = plt.figure()
    # separating the columns to plot
    data_1 = train_extracted_features[:, 0]
    data_2 = train_extracted_features[:, 1]
    # plotting the data as points
    plt.plot(data_1, data_2, 'o')
    plt.title('Extracted Features using PCA')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    fig.savefig('Extracted Features.png')

    color_indices = cluster_Kmeans(train_extracted_features)

    fig = plt.figure()
    plt.scatter(data_1, data_2, c=color_indices)
    plt.title('Clustered Features using KMeans')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    fig.savefig('Clustered Features.png')

    # plt.show()

    # Finished training
    # repeating for test data

    Test_filtered = BFP(test_data, Test_t, dt)

    NEO_filtered = NEO_function(Test_filtered)

    var = statistics.median(np.abs(NEO_filtered)) / 0.67
    k = 4
    threshold = var * k

    AP = Align_peaks(dt, fs, NEO_filtered, Test_filtered)

    test_extracted_features = PCA_analysis(AP)

    color_indices = cluster_Kmeans(test_extracted_features)

    # print(color_indices)
    neuron_1, neuron_2, neuron_3 = cluster_assign(
        train_extracted_features, test_extracted_features, Test_t, color_indices, fs)

    fig = plt.figure()
    plt.plot(Test_t, neuron_1)
    plt.title('Spikes - Neuron 1')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spikes')
    fig.savefig('neuron_1.png')

    fig = plt.figure()
    plt.plot(Test_t, neuron_2)
    plt.title('Spikes - Neuron 2')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spikes')
    fig.savefig('neuron_2.png')

    fig = plt.figure()
    plt.plot(Test_t, neuron_3)
    plt.title('Spikes - Neuron 3')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spikes')
    fig.savefig('neuron_3.png')

    plt.show()
