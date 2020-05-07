# Final-Project

This is a group project for Software Carpentry EN.540.635 to extract neural action potentials from electrode recordings. Group members and their emails: Molly Acord, macord1@jhu.edu Sreelakshmi Sunil, ssunil1@jhu.edu

Necessary files included in Final Release v.f : Final.py and NII_Data.csv

Final.py is the master code for this project.

Method of solving: separate data in 5 seconds of traing data and 20 seconds of testing data, bandpass filter the raw data, apply nonlinear energy operator and threshold, align action potentials (or spikes), extract features using Principle Component Analysis, cluster features using k-MEANS, and classify spikes in testing data to their respective neuron by finding k Nearest Neighbors in training data. 

Results of this program can bee seen in figures:

Filtered Training Data.png
Filtered Testing Data.png
NEO Training Data.png
NEO Testing Data.png
Aligned Peaks of Training data.png
Aligned Peaks of Testing data.png
Extracted Features Train_data.png
Clustered Features Train_data.png
neuron_1.png
neuron_2.png
neuron_3.png

The neuron figures show approximately how many spikes belong to each neuron during a 1 second period of the testing data.
