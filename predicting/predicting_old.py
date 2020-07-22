import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
import csv
import json

model = keras.models.load_model('my_model')

with open("predicting_test.csv", "w", encoding="utf-8", newline='') as outputFilePredict:
    # json.dump(listOfGames, outputFile)
    writer = csv.writer(outputFilePredict)
    writer.writerow(['original_price', 'positive_review', 'num_of_movies',
                     'num_of_screenshots', 'game_description_len', "RPG", "Action", "Adventure", "Casual",
                     "Indie", "Massively Multiplayer", "Racing",
                     "Simulation", "Sports", "Strategy", 'num_of_languages'])
    writer.writerow([59.99, 98, 13, 11, 2000, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 16])

outputFilePredict.close()

with open('mean_and_std.json', 'r') as outputMeanStd:
    mean_and_std = json.load(outputMeanStd)

outputMeanStd.close()

data = pd.read_csv('predicting_test.csv')
# game_info = pd.Series(np.array([59.99, 98, 13, 11, 2000, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 16]))

'''game_info = np.array([[59.99], [98], [13], [11], [2000], [1], [1], [1], [0], [0], [0], [0], [0], [0],
                     [0], [16]])'''

print(data.shape)
prediction = model.predict(x=data)

print(prediction * mean_and_std['std'] + mean_and_std['mean'])