import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math

data_url = f'final_data.csv'
data = pd.read_csv(data_url)

data = data.drop('total_points_next', axis=1)


def split_dataset(dataset, test_ratio=.3):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


train_ds_pd, test_ds_pd = split_dataset(data)

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    train_ds_pd, label='element_type')
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_ds_pd, label='element_type')

model_1 = tfdf.keras.RandomForestModel(verbose=2)
model_1.fit(train_ds)
model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds)
print()

# Save the model
model_1.save("random_forest_model")

# Load the model
loaded_model = tf.keras.models.load_model("random_forest_model")
