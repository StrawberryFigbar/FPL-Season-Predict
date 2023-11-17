import pandas as pd
import tensorflow as tf
import numpy as np


Data2022URL = f'Data/players_raw_2022.csv'
Data2022 = pd.read_csv(Data2022URL)
Data2022 = Data2022[Data2022['news'].isnull()]
names2022 = Data2022[['first_name',
                      'second_name', 'element_type', 'team']].copy()
Data2022 = Data2022[['assists', 'element_type', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored',
                     'minutes', 'influence', 'points_per_game', 'saves', 'bonus', 'yellow_cards', 'total_points']]
Data2022 = Data2022.astype(int)

tf.convert_to_tensor(Data2022)
normalizer = tf.keras.layers.Normalization(axis=-1)


def get_basic_model():
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(275)
    ])
    return model


model = get_basic_model()
checkpoint_path = "training_1/cp.ckpt"
model.load_weights(checkpoint_path).expect_partial()
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
predictions = probability_model.predict(Data2022)
names2022['prediction'] = predictions.argmax(axis=1)
names = names2022.sort_values('element_type').sort_values('prediction')
print(names.tail(10))
