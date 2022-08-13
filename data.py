import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from scipy import signal
import scipy


DATA_DIR = "dataset/"
class_names = ['left', 'right', 'front', 'back']
class_names_label = {
	'left': 0,
	'right': 1,
	'front': 2,
	'back': 3
}
nb_classes = 4


# Butter filter params
# fs = 17000
# fc = 50
# t = np.arange(17000) / fs
# w = fc / (fs / 2)
# b, a = scipy.signal.butter(5, w, 'low')

def load_data():
	datasets = ['train']

	for dataset in datasets:
		directory = DATA_DIR + dataset
		csv = []
		labels = []
		for folder in os.listdir(directory):
			curr_label = class_names_label[folder]
			for file in os.listdir(os.path.join(directory, folder)):
				X_path = os.path.join(directory, folder, file)
				X_csv = pd.read_csv(X_path)
				X_csv.drop('Time', inplace=True, axis=1)
				X_csv[~X_csv.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)
				# X_csv = X_csv.iloc[:, 0:23]
				value = X_csv.shape[0] - 17000
				X_csv = X_csv.iloc[:-value]
				# for channel in X_csv:
				# 	signal = np.array(X_csv[channel])
				# 	X_csv[channel] = scipy.signal.filtfilt(b, a, signal)
				X_csv = X_csv.to_numpy()
				csv.append(X_csv)
				labels.append(curr_label)
			sample = np.array(csv)
			data = np.dstack(sample)
		labels = np.array(labels)

	return data, labels

# X, y = load_data()
# print("Training examples shape: " + str(X.transpose().shape))
# print("Training labels shape: " + str(y.shape))

# print(np.isinf(X).any())

# print("Testing examples shape: " + str(test_X.shape))
# print("Testing labels shape: " + str(test_y.shape))