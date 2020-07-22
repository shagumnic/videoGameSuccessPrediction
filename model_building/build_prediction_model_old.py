import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import text, sequence
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import json
import re

# The following lines adjust the granularity of reporting.
'''pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format'''

# data = pd.read_csv('dataForMachineLearning.csv')
data = pd.read_csv('dataForRecommendation.csv')
data_df = data[['original_price', 'genres', 'positive_review',
                'num_of_movies', 'num_of_screenshots', 'game_description_len',
                'languages', 'num_players_after_month']]

print(data_df.shape)
# normalized and encode the data here

# data_df['positive_review'] /= 100.0
# data_df['genres'] = [data_df['genres'][i]['description'] for i in range(len(data_df['genres']))]
'''for language in ['English', 'German', 'French', 'Spanish', 'Italian', 'Russian',
                 'Japanese', 'Portuguese-Brazil', 'Polish', 'Simplified Chinese']:
    data_df.at[:, language] = 0'''

for genre in ("RPG", "Action", "Adventure", "Casual",
              "Indie", "Massively Multiplayer", "Racing",
              "Simulation", "Sports", "Strategy"):
    data_df.loc[:, genre] = 0

data_df.loc[:, 'num_of_languages'] = 0

data_df['languages'] = data_df['languages'].map(lambda x: x[1:len(x)-1].replace("'", "").split(', '))

data_df['genres'] = data_df['genres'].map(lambda x: x[1:len(x)-1].replace("'", "").split(', '))

'''data_df['genres'] = data_df['genres'].map({"RPG":"RPG", "Action":"Action", "Adventure":"Adventure",
                                           "Casual":"Casual", "Indie":"Indie",
                                           "Massively Multiplayer":"Massively Multiplayer", "Racing": "Racing",
                                          "Simulation":"Simulation", "Sports":"Sports", "Strategy":"Strategy"})'''

for index, row in data_df.iterrows():
    # print(len(languages_list))
    if "RPG" in row['genres']:
        data_df.loc[index, 'RPG'] = 1

    if "Action" in row['genres']:
        data_df.loc[index, 'Action'] = 1

    if "Adventure" in row['genres']:
        data_df.loc[index, 'Adventure'] = 1

    if "Casual" in row['genres']:
        data_df.loc[index, 'Casual'] = 1

    if "Indie" in row['genres']:
        data_df.loc[index, 'Indie'] = 1

    if "Massively Multiplayer" in row['genres']:
        data_df.loc[index, 'Massively Multiplayer'] = 1

    if "Racing" in row['genres']:
        data_df.loc[index, 'Racing'] = 1

    if "Simulation" in row['genres']:
        data_df.loc[index, 'Simulation'] = 1

    if "Sports" in row['genres']:
        data_df.loc[index, 'Sports'] = 1

    if "Strategy" in row['genres']:
        data_df.loc[index, 'Strategy'] = 1

    data_df.loc[index, 'num_of_languages'] = len(row['languages'])

    # vector_matrix[index, :]

print(data_df.loc[0, :])
# data_df = pd.concat([data_df, pd.get_dummies(data_df['languages'], prefix='language')], axis=1)

data_df = data_df.drop(['languages'], axis=1)

data_df = data_df.drop(['genres'], axis=1)

print(data_df.mean())
print(data_df.std())
mean_and_std = {'mean': data_df.mean().num_players_after_month, 'std': data_df.std().num_players_after_month}

with open('mean_and_std.json', 'w') as outputMeanStd:
    json.dump(mean_and_std, outputMeanStd)

outputMeanStd.close()
data_df = (data_df - data_df.mean()) / data_df.std()
# randomize and split data into train set and test set
data_df = data_df.reindex(np.random.permutation(data_df.index))

train_data_df = data_df.sample(frac=0.8, random_state=0)
test_data_df = data_df.drop(train_data_df.index)

# inspect data
sns.pairplot(train_data_df[['original_price', 'positive_review', 'num_of_movies', 'num_of_screenshots',
                            'game_description_len', 'num_of_languages', 'num_players_after_month']], diag_kind='kde')
# overall statistic
train_stats = train_data_df.describe()
train_stats.pop('num_players_after_month')
train_stats = train_stats.transpose()

# split features from labels
train_labels = train_data_df.pop('num_players_after_month')
test_labels = test_data_df.pop('num_players_after_month')

# train_labels /= 100
# test_labels /= 100


# z normalized, use train stats for both train and test cause need to be the same distribution
'''def z_normalized(data):
    return (data - data.mean()) / data.std()


normed_train_data = z_normalized(train_data_df)
normed_test_data = z_normalized(test_data_df)
normed_train_labels = z_normalized(train_labels)
normed_test_labels = z_normalized(test_labels)'''

print(train_data_df.loc[0, :])
# print(normed_train_data.loc[0, :])
# print(normed_train_data.tail())
# build model

def build_linear_model(my_learning_rate):
    model = keras.models.Sequential()

    # model.add(feature_layer)

    model.add(keras.layers.Dense(units=1, input_shape=[len(train_data_df.keys())]))

    # model.add(keras.layers.Dense(units=64, activation='relu'))

    model.add(keras.layers.Dense(units=1, name='Output'))

    optimizer = keras.optimizers.RMSprop(lr=my_learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model


def train_linear_model(model, data_set, train_labels, epochs, batch_size, callbacks):
    history = model.fit(data_set, train_labels, batch_size=batch_size, epochs=epochs,
                        validation_split=0.2, shuffle=True) #callbacks=callbacks)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist['mse']
    mae = hist['mae']

    return epochs, rmse, mae, history


def plot_loss_curve(epochs, mse):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')

    plt.plot(epochs, mse, label='Loss')
    plt.legend()
    plt.ylim([mse.min()*0.95, mse.max()*1.03])
    plt.show()


def build_deep_model(my_learning_rate):
    model = keras.models.Sequential()

    model.add(layers.Dense(units=64, activation='relu', input_shape=[len(train_data_df.keys())]))


    model.add(layers.Dense(units=64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.04),
                                 name='Hidden1'))

    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(units=32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.04),
                                 name='Hidden2'))

    model.add(layers.Dense(units=1, name='Output'))

    optimizer = keras.optimizers.Adam(lr=my_learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model


def train_deep_model(model, data_set, train_labels, epochs, batch_size, callbacks) :
    history = model.fit(x=data_set, y=train_labels, epochs=epochs, batch_size=batch_size, callbacks=callbacks, shuffle=True)

    # history = model.fit(x=data_set, y=train_labels, epochs=epochs, callbacks=callbacks, shuffle=True)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist['mse']
    mae = hist['mae']

    return epochs, mse, mae, history


# start the implementing process here
learning_rate = 0.03
epochs = 1000
batch_size = 2000


def linear_regression(normed_train_data, train_labels, normed_test_data, test_labels, learning_rate, epochs, batch_size):

    my_linear_model = build_linear_model(learning_rate)

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    linear_epochs, linear_mse, linear_mae, linear_history = train_linear_model(my_linear_model, normed_train_data, train_labels, epochs, batch_size, [early_stop])

    plot_loss_curve(linear_epochs, linear_mse)

    my_linear_model.evaluate(normed_test_data, test_labels, batch_size=batch_size)

    save_model(my_linear_model)


def deep_linear(normed_train_data, train_labels, normed_test_data, test_labels, learning_rate, epochs, batch_size) :

    my_deep_model = build_deep_model(learning_rate)

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    deep_epochs, deep_mse, deep_mae, deep_history = train_deep_model(my_deep_model, normed_train_data, train_labels, epochs, batch_size, [early_stop])

    plot_loss_curve(deep_epochs, deep_mse)

    loss, mae, mse = my_deep_model.evaluate(normed_test_data, test_labels, batch_size=batch_size, verbose=1)

    test_predictions = my_deep_model.predict(normed_test_data).flatten()

    # plt.figure()

    a = plt.axes(aspect='equal')

    plt.scatter(test_labels, test_predictions)

    plt.xlabel('True Values [num_of_players]')
    plt.ylabel('Prediction [num_of_players]')

    lims = [0, 10]

    # plt.plot(test_labels, test_predictions, label='Comparison')
    # plt.legend()
    plt.xlim([test_labels.min() * 0.95, test_labels.max() * 1.03])
    plt.ylim([test_predictions.min() * 0.95, test_predictions.max() * 1.03])

    _ = plt.plot([test_labels.min() * 0.95, test_labels.max() * 1.03],
                 [test_predictions.min() * 0.95, test_predictions.max() * 1.03])
    plt.show()
    save_model(my_deep_model)


def save_model(model):
    choice = input('Do you want to save this model? (y/n) ')
    if choice == 'y':
        model.save('my_model')
    elif choice == 'n':
        pass
    else:
        print('Your input is invalid')


option = input('Please enter the prediction system you wanna use (linear regression or neural network (l/n): ')

if option == 'l':
    # linear_regression(normed_train_data, normed_train_labels, normed_test_data, normed_test_labels, learning_rate, epochs, batch_size)

    linear_regression(train_data_df, train_labels, test_data_df, test_labels, learning_rate,
                      epochs, batch_size)

elif option == 'n':
    # deep_linear(normed_train_data, normed_train_labels, normed_test_data, normed_test_labels, learning_rate, epochs, batch_size)

    deep_linear(train_data_df, train_labels, test_data_df, test_labels, learning_rate,
                      epochs, batch_size)

else:
    print('Your input is not valid')
