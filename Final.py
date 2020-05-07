'''
EN.540.635 Software Carpentry
Final Project Spike Detection
Molly Acord and Sreelakshmi Sunil
'''

import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance
import copy


def Read_file(filename, fs, train_time, total_time):
    '''
    This will read in the data file and separate the data based on
    desired training time, total time, and frequency sampling

    **Parameters**

        filename: *str*
            Data file
        fs: *float*
            Sampling frequency
        train_time: *int*
            How many seconds of the data are assigned to training the data
        total_time: *int*
            Time in seconds of total data being used

    **Returns**

        dt: *float*
            Time difference between each each data point
        train_data: *list, float*
            Volatge values recorded at each time point during training period
        test_data: *list, float*
            Volatge values recorded at each time point during testing period
        Train_t: *list, float*
            Time values for each recorded data point in training period
        Test_t: *list, float*
            Time values for each recorded data point in testing period
    '''
    file = open(filename)
    line = next(file).strip('\n').split(',')
    dt = 1 / fs
    train_data = []
    test_data = []
    for i in range(0, int(train_time * fs)):
        train_data.append(float(line[i]))
    for i in range(int(train_time * fs) + 1, len(line)):
        test_data.append(float(line[i]))
    Train_t = np.arange(0.0, float(train_time) - dt, dt)
    Test_t = np.arange(5.0, float(total_time), dt)

    return dt, train_data, test_data, Train_t, Test_t


def BFP(data, time, dt):
    '''
    This will filter data using a band pass filter for frequencies
    between 300Hz and 3000Hz. This the the common range for neural recordings.

    **Parameters**

        data: *list, float*
            Volatge values recorded at each time point during specified
            time period
        time: *list, float*
            Time values for each recorded data point
        dt: *float*
            Time difference between each each data point

    **Returns**

        Vout_2: *list, float*
            Volatge values recorded at each time point that were allowed
            through bandpass filter
    '''
    # High pass filter
    Vin_1 = data
    Vin_prev = 0
    Vout = 0
    Vout_1 = []
    RC = 5.3E-4  # for a cuttoff frequency of 300 Hz
    for i in range(0, len(time) - 1):
        dVin = Vin_1[i] - Vin_prev
        dVout = dVin - dt * Vout / RC
        Vout = Vout + dVout
        Vout_1.append(Vout)
        Vin_prev = Vin_1[i]
    # Low pass filter
    Vin_2 = Vout_1
    Vout = 0
    dVout = 0
    Vout_2 = [0]
    RC = 5.3E-5  # for a cuttoff frequency of 3000 Hz
    for i in range(0, len(time) - 1):
        dVout = dt * ((Vin_2[i] - Vout) / RC)
        Vout = Vout + dVout
        Vout_2.append(Vout)
    return Vout_2


def NEO_function(data):
    '''
    This will calculate the energy of a signal, making it easier
    for spike identification later.

    **Parameters**

        data: *list, float*
            Volatge values recorded at each time point that were allowed
            through bandpass filter

    **Returns**

        NEO: *list, float*
            Nonlinear energy operator - basically the energy of a signal
    '''
    NEO = []
    for i in range(0, len(data)):
        if i == 0:
            # NEO equation for start point of data
            neo = data[i] ** 2 - data[i + 1]
        elif i == len(data) - 1:
            # NEO quation for end point of data
            neo = data[i] ** 2 - data[i - 1]
        else:
            # NEO main equation
            neo = data[i] ** 2 - (data[i + 1] * data[i - 1])
        NEO.append(neo)
    return NEO


def Align_peaks(dt, fs, NEO, Filtered_data, fig_name):
    '''
    This will detect spikes using a threshold and align them at the maximum
    during the 3ms window in which an action potential fires.

    **Parameters**

        dt: *float*
            Time difference between each each data point
        fs: *float*
            Sampling frequency
        NEO: *list, float*
            Nonlinear energy operator - basically the energy of a signal
        Filtered_data: *list, float*
            Volatge values recorded at each time point that were allowed
            through bandpass filter
        fig_name: *str*
            Desired name of figure

    **Returns**

        APs: *list, float*
            Voltage values of action potential, or peaks
    '''

    pts_before = round(0.001 * fs)  # 1ms before peak of AP
    pts_after = round(0.002 * fs)  # 2ms after peak of AP

    # 3ms in which an AP fires and returns to resting potential
    t = np.arange(0, 0.003, dt)

    refractory = 2E-3  # 2ms refractory period after an AP fires
    refrac_pts = int(refractory * fs)
    APs = []
    i = 0
    fig = plt.figure()
    while i <= len(NEO) - refrac_pts:
        peak = []
        if NEO[i] >= threshold:
            # create window around refractory period
            window = [i, i + refrac_pts]
            peak = Filtered_data[window[0]:window[1]]
            max_value = max(peak)  # find max value in window to mark as peak
            max_index = peak.index(max_value)  # find index of the peak
            max_index = max_index + i - 1
            # create a window of total AP points (1ms before peak and
            # 2ms after peak)
            window2 = [max_index - pts_before, max_index + pts_after]
            # collect filtered voltage values of the peak
            filt_window = Filtered_data[window2[0]:window2[1]]
            plt.plot(t, filt_window)
            APs.append(filt_window)  # save volateg values of each peak
            i = i + refrac_pts
        else:
            i = i + 1

    plt.xlim(0, 0.003)  # 3ms window for AP firing
    plt.title('Aligned Action Potentials')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    fig.savefig(fig_name)

    return(APs)


def PCA_analysis(AP):
    '''
    This will extract features of action potentail data
    using PCA analysis.

    **Parameters**

        AP: *list, float*
            Voltage values of action potential, or peaks

    **Returns**

        transformed_data: *list, float*
            Principle components / first two dimentions
            describing location of action potentaisl
    '''
    # setting PCA features as 2
    pca = PCA(n_components=2)
    # transforming the data
    transformed_data = pca.fit_transform(AP)

    return(transformed_data)


def cluster_Kmeans(data):
    '''
    Clusters data based on KMeans algorithm

    ***Parameters***
        data : output matrix of PCA analysis size r x 2 
               data contains extracted features

    ***Returns***

        color_indices(array): consists of 0, 1, 2
            depending on which cluster it is assigned to
    '''
    # each cluster represent 1 neuron
    # using KMeans function to identify clusters
    CLUSTER = KMeans(n_clusters=3).fit(data)

    # assigning same integer values to data of same cluster
    color_indices = CLUSTER.predict(data)

    return(color_indices)


def cluster_assign(train_PCA, test_PCA, test_time, index, fs):
    '''
    Assigns spikes in test data to any of the three neurons
    3 clusters - 3 neurons
    Based on k nearest neighbours algorithm

    ***Parameters***
        train_PCA : r x 2 matrix containing
                     extracted features of the training data
        test_PCA :  r x 2 matrix containing
                     extracted features of the test data
        test_time : time distribution array for test data
        index : color indices array output of KMeans fuction
                for train data
        fs(int) : sampling frequency

    ***Returns***
        Neuron_1(array): neuronal spike data for neuron 1
        Neuron_2(array): neuronal spike data for neuron 2
        Neuron_3(array): neuronal spike data for neuron 3

    '''
    # considers 10 nearest neighbours
    k = 11
    dur = 0
    refractory_pd = round(0.002 * fs)
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
            temp = distance.euclidean(train_PCA[j], test_PCA[i])
            d.append(temp)

        # storing indices of k nearest neighbours
        idx = np.argpartition(d, k)[:k]

        # identifying which cluster it belongs to
        for j in range(0, k):
            cluster_check.append(index[idx[j]])

        # looks for cluster with max occurence
        # and assigns to specific neuron
        # 0 - neuron_1, 1 - neuron_2 and 2 - neuron_3
        try:
            check = statistics.mode(cluster_check)

        except statistics.StatisticsError:
            # error occurs when there is 2 mode values
            # code to calculate mode
            # uses statistics.mode() otherwise to increase speed
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
            check = np.argpartition(c, -1)[-1:]

        # checking which cluster it belongs to
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
            print("out of neurons")
            continue

    return (neuron_1, neuron_2, neuron_3)


if __name__ == "__main__":
    print('Please wait. This may take a few minutes.')
    fs = 2.23221E4  # sampling frequency
    train_time = 5  # seconds of data used for training
    total_time = 25  # seconds of total data used
    dt, train_data, test_data, Train_t, Test_t = Read_file(
        'NII_Data.csv', fs, train_time, total_time)

    Train_filtered = BFP(train_data, Train_t, dt)

    fig = plt.figure()
    plt.plot(Train_t, Train_filtered)
    plt.xlim(0, 1)
    plt.title('1 second of Filtered Training Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    fig.savefig('Filtered Training Data.png')

    NEO_filtered = NEO_function(Train_filtered)

    fig = plt.figure()
    plt.plot(Train_t, NEO_filtered)
    plt.xlim(0, 1)
    plt.title('1 second of NEO Training Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Nonlinear Energy Operator')
    fig.savefig('NEO Training Data.png')

    var = statistics.median(np.abs(NEO_filtered)) / 0.67
    k = 4
    threshold = var * k

    AP_train = Align_peaks(dt, fs, NEO_filtered, Train_filtered,
                           'Aligned Peaks of Training data.png')

    train_extracted_features = PCA_analysis(AP_train)

    fig = plt.figure()
    # separating the columns to plot
    data_1 = train_extracted_features[:, 0]
    data_2 = train_extracted_features[:, 1]
    # plotting the data as points
    plt.plot(data_1, data_2, 'o')
    plt.title('Extracted Features of Training data using PCA')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    fig.savefig('Extracted Features Train_data.png')

    color_indices = cluster_Kmeans(train_extracted_features)
    index = copy.deepcopy(color_indices)

    fig = plt.figure()
    plt.scatter(data_1, data_2, c=color_indices)
    plt.title('Clustered Features of Training data using KMeans')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    fig.savefig('Clustered Features Train_data.png')

    # TRAINING DATA FINISHED
    # repeat for test data

    Test_filtered = BFP(test_data, Test_t, dt)

    fig = plt.figure()
    plt.plot(Test_t, Test_filtered)
    plt.xlim(5, 6)
    plt.title('1 second of Filtered Testing Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    fig.savefig('Filtered Testing Data.png')

    NEO_filtered = NEO_function(Test_filtered)

    fig = plt.figure()
    plt.plot(Test_t, NEO_filtered)
    plt.xlim(5, 6)
    plt.title('1 second of NEO Training Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Nonlinear Energy Operator')
    fig.savefig('NEO Testing Data.png')

    var = statistics.median(np.abs(NEO_filtered)) / 0.67
    k = 4
    threshold = var * k

    AP_test = Align_peaks(dt, fs, NEO_filtered, Test_filtered,
                          'Aligned Peaks of Testing data.png')

    test_extracted_features = PCA_analysis(AP_test)

    neuron_1, neuron_2, neuron_3 = cluster_assign(
        AP_train, AP_test, Test_t, index, fs)

    fig = plt.figure()
    plt.plot(Test_t, neuron_1)
    plt.xlim(5, 6)
    plt.title('Spikes - Neuron 1')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spikes')
    fig.savefig('neuron_1.png')

    fig = plt.figure()
    plt.plot(Test_t, neuron_2)
    plt.xlim(5, 6)
    plt.title('Spikes - Neuron 2')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spikes')
    fig.savefig('neuron_2.png')

    fig = plt.figure()
    plt.plot(Test_t, neuron_3)
    plt.xlim(5, 6)
    plt.title('Spikes - Neuron 3')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spikes')
    fig.savefig('neuron_3.png')

    print('\n')
    print('PROGRAM HAS FINISHED')
    print('\n')
    print('Please see the following images for results: ')
    print('\n')
    print('Filtered Training Data.png')
    print('Filtered Testing Data.png')
    print('NEO Training Data.png')
    print('NEO Testing Data.png')
    print('Aligned Peaking of Training data.png')
    print('Aligned Peaking of Testing data.png')
    print('Extracted Features Train_data.png')
    print('Clustered Features Train_data.png')
    print('neuron_1.png')
    print('neuron_2.png')
    print('neuron_3.png')

    plt.show()
