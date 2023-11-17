import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math

data_url = f'final_data.csv'
data = pd.read_csv(data_url)


def split_dataset(dataset, test_ratio=.3):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


train_ds_pd, test_ds_pd = split_dataset(data)
print(len(train_ds_pd), len(test_ds_pd))
# Calculate the minimum and maximum values of 'total_points_next'
min_points = data['total_points_next'].min()
max_points = data['total_points_next'].max()

# Define a function to scale 'total_points_next' to the range 1-100


def scale_points(value):
    return int(1 + ((value - min_points) / (max_points - min_points) * 99))


# Apply this function to the 'total_points_next' column in both datasets
train_ds_pd['total_points_next'] = train_ds_pd['total_points_next'].apply(
    scale_points)
test_ds_pd['total_points_next'] = test_ds_pd['total_points_next'].apply(
    scale_points)

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    train_ds_pd, label='total_points_next')
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_ds_pd, label='total_points_next')

model_1 = tfdf.keras.RandomForestModel(verbose=2)
model_1.fit(train_ds)
model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds)
print()

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")
