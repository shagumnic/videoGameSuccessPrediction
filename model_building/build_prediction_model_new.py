import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from rake_nltk import Rake
import re

data_df = pd.read_csv('dataForRecommendation.csv', encoding='utf-8')
data_df = data_df[['original_price', 'genres', 'positive_review', 'developers',
                   'num_of_movies', 'num_of_screenshots', 'game_description_len',
                   'languages', 'num_players_after_month', 'description']]


# normalized and encode the data here
numerical_columns = ['positive_review', 'num_of_movies', 'num_of_screenshots',
                     'game_description_len', 'num_players_after_month', "RPG", "Action",
                     "Adventure", "Casual", "Indie", "MassivelyMultiplayer", "Racing",
                     "Simulation", "Sports", "Strategy", 'num_of_languages']

bucket_numerical_columns = ['original_price']

embedding_columns = ['bags_of_words', 'developers_bags', 'languages_bags']


# load the pre-trained word-embedding vectors
feature_columns = []
regex = re.compile(r'[\n\r\t\xa0]')

data_df['description'] = data_df['description'].map(lambda x: regex.sub(" ", x)[16:])
data_df['developers'] = data_df['developers'].map(lambda x: x[1:len(x)-1].replace("'", "").split(', '))

for genre in ("RPG", "Action", "Adventure", "Casual",
              "Indie", "MassivelyMultiplayer", "Racing",
              "Simulation", "Sports", "Strategy"):
    data_df[genre] = 0

data_df['num_of_languages'] = 0

data_df['languages'] = data_df['languages'].map(lambda x: x[1:len(x)-1].replace("'", "").split(', '))

data_df['genres'] = data_df['genres'].map(lambda x: x[1:len(x)-1].replace("'", "").split(', '))

compatible_regex = re.compile(r'[ -]')
languagesList = []
with open('languagesList.txt', 'r') as inputLanguageFile:
    for language in inputLanguageFile:
        languageConvert = regex.sub("", language)
        languagesList.append(languageConvert)
        if language != 'English\n':
            numerical_columns.append(compatible_regex.sub("", languageConvert))

inputLanguageFile.close()


def get_keys(data_text):
    r = Rake()

    r.extract_keywords_from_text(data_text)

    key_words_dict_scores = r.get_word_degrees()

    keys = list(key_words_dict_scores.keys())

    return keys


data_df['Key_words'] = ''

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
        data_df.loc[index, 'MassivelyMultiplayer'] = 1

    if "Racing" in row['genres']:
        data_df.loc[index, 'Racing'] = 1

    if "Simulation" in row['genres']:
        data_df.loc[index, 'Simulation'] = 1

    if "Sports" in row['genres']:
        data_df.loc[index, 'Sports'] = 1

    if "Strategy" in row['genres']:
        data_df.loc[index, 'Strategy'] = 1

    data_df.loc[index, 'num_of_languages'] = len(row['languages'])

    for language in languagesList:
        languageCompatible = compatible_regex.sub("", language)
        if language in row['languages']:
            data_df.at[index, languageCompatible] = 1
        else:
            data_df.at[index, languageCompatible] = 0
    description = row['description']
    key_words = get_keys(description)
    data_df.at[index, 'Key_words'] = key_words

data_df.drop(columns=['description', 'English'], inplace=True)

data_df['bags_of_words'] = ''

data_df['developers_bags'] = ''

data_df['languages_bags'] = ''

columns = data_df.columns

for index, row in data_df.iterrows():
    bags_of_words = ''
    bags_of_words = bags_of_words + ' '.join(row['Key_words']) + ' '
    data_df.at[index, 'bags_of_words'] = bags_of_words

    developers_bags = ''
    developers_bags = developers_bags + ' '.join(row['developers']) + ' '
    data_df.at[index, 'developers_bags'] = developers_bags


developers_vocab = data_df['developers_bags'].unique()
developers = tf.feature_column.categorical_column_with_vocabulary_list('developers_bags',
                                                                       vocabulary_list=developers_vocab)
developers = tf.feature_column.embedding_column(developers, 50)

feature_columns.append(developers)

description = tf.feature_column.categorical_column_with_hash_bucket('bags_of_words',
                                                                    hash_bucket_size=1200)
description = tf.feature_column.embedding_column(description, 50)

feature_columns.append(description)

data_df = data_df.drop(['languages', 'genres', 'Key_words', 'developers'], axis=1)

# randomize and split data into train set and test set
data_df = data_df.reindex(np.random.permutation(data_df.index))

train_data_df = data_df.sample(frac=0.8, random_state=0)
test_data_df = data_df.drop(train_data_df.index)


def normalized_data(data_feature_column):
    return (data_feature_column - data_feature_column.mean())/data_feature_column.std()


train_data_norm = train_data_df.copy()

test_data_norm = test_data_df.copy()
for column in numerical_columns:
    if column != 'num_players_after_month':
        feature_columns.append(tf.feature_column.numeric_column(column))
    train_data_norm[column] = normalized_data(train_data_df[column])
    test_data_norm[column] = normalized_data(test_data_df[column])


price = tf.feature_column.numeric_column('original_price')

price_buckets = tf.feature_column.bucketized_column(price, boundaries=[10, 20, 40, 60])

feature_columns.append(price_buckets)

feature_layers = layers.DenseFeatures(feature_columns)

# inspect data
sns.pairplot(train_data_df[['original_price', 'positive_review', 'num_of_movies', 'num_of_screenshots',
                            'game_description_len', 'num_of_languages', 'num_players_after_month']],
             diag_kind='kde')
# overall statistic
train_stats = train_data_df.describe()
train_stats.pop('num_players_after_month')
train_stats = train_stats.transpose()


# build model
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

    model.add(feature_layers)

    model.add(layers.Dense(units=64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.04),
                           name='Hidden1'))

    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(units=32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.04),
                           name='Hidden2'))

    model.add(layers.Dense(units=1, name='Output'))

    optimizer = keras.optimizers.Adam(lr=my_learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model


def train_deep_model(model, train_data_norm, label_name, epochs, batch_size, callbacks):
    features = {name: np.array(value) for name, value in train_data_norm.items()}
    labels = np.array(features.pop(label_name))
    history = model.fit(x=features, y=labels, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=callbacks, shuffle=True)

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
label_name = 'num_players_after_month'


def deep_linear(train_data_norm, label_name, test_data_norm, learning_rate, epochs, batch_size) :

    my_deep_model = build_deep_model(learning_rate)

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    deep_epochs, deep_mse, deep_mae, deep_history = train_deep_model(my_deep_model,
                                                                     train_data_norm,
                                                                     label_name,
                                                                     epochs, batch_size, [early_stop])

    plot_loss_curve(deep_epochs, deep_mse)

    features = {name: np.array(value) for name, value in test_data_norm.items()}
    labels = np.array(features.pop(label_name))

    my_deep_model.evaluate(features, labels, batch_size=batch_size, verbose=1)

    test_predictions = my_deep_model.predict(features).flatten()

    a = plt.axes(aspect='equal')

    plt.scatter(labels, test_predictions)

    plt.xlabel('True Values [num_of_players]')
    plt.ylabel('Prediction [num_of_players]')

    plt.xlim([labels.min() * 0.95, labels.max() * 1.03])
    plt.ylim([test_predictions.min() * 0.95, test_predictions.max() * 1.03])

    _ = plt.plot([labels.min() * 0.95, labels.max() * 1.03],
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


deep_linear(train_data_norm, label_name, test_data_norm, learning_rate,
            epochs, batch_size)

