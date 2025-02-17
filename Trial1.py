import numpy as np
import matplotlib.pyplot as plt
import statistics


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
