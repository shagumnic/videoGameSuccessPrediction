from tkinter import messagebox
import numpy as np
from tensorflow.keras import models
from rake_nltk import Rake
from tkinter import *
import re

model = models.load_model('my_model')

root = Tk()
root.title('Video Game Succession Prediction')
root.maxsize(1280, 1024)

Input_frame = Frame(root, width=1280, height=1024, bg='grey')
Input_frame.grid(row=0, column=0)

game_predicted = {}


def predict():
    global game_predicted
    global languagesVarList
    global genresVarList
    try:
        game_predicted['original_price'] = np.array([float(price.get())])
        game_predicted['positive_review'] = np.array([float(positive_review.get())])
        game_predicted['num_of_movies'] = np.array([float(num_of_movies.get())])
        game_predicted['num_of_screenshots'] = np.array([float(num_of_screenshots.get())])
    except ValueError:
        messagebox.showwarning('Missing or Invalid Input',
                               'You haven\'t filled out all the required information '
                               'or what you type in is not valid')
    else:
        language_count = 0
        for language_var_key, language_var_value in languagesVarList.items():
            language_compatible = language_var_value.get()
            if language_compatible:
                if language_var_key != 'English':
                    game_predicted[language_var_key] = np.array([1])
                language_count += 1
            else:
                game_predicted[language_var_key] = np.array([0])
        for genre_var_key, genre_var_value in genresVarList.items():
            genre_compatible = genre_var_value.get()
            if genre_compatible:
                game_predicted[genre_var_key] = np.array([1])
            else:
                game_predicted[genre_var_key] = np.array([0])
        game_predicted['num_of_languages'] = np.array([float(language_count)])
        description_var = description.get("1.0", 'end-1c')
        game_predicted['game_description_len'] = np.array([len(description_var)])
        r = Rake()
        r.extract_keywords_from_text(description_var)
        key_words_dict_scores = r.get_word_degrees()
        key_words = list(key_words_dict_scores.keys())
        game_predicted['bags_of_words'] = np.array([' '.join(key_words) + ' '])
        developers_var = developers.get().split(",")
        game_predicted['developers_bags'] = np.array([' '.join(developers_var) + ' '])
        prediction_result = model.predict(x=game_predicted)
        final_result = prediction_result[0][0]
        result_str = "The game " + name.get() + " will have on average about " + str(final_result)\
                     + " players after a month from released."
        messagebox.showinfo('Result', result_str)


Label(Input_frame, text='Game name', bg='grey').grid(row=0, column=0, padx=5, pady=5, sticky=W)
name = Entry(Input_frame)
name.grid(row=0, column=1, padx=5, pady=5, sticky=W)

Label(Input_frame, text='Price($)', bg='grey').grid(row=1, column=0, padx=5, pady=5, sticky=W)
price = Entry(Input_frame)
price.grid(row=1, column=1, padx=5, pady=5, sticky=W)

Label(Input_frame, text='Positive Critics Review Percentage(%)', bg='grey').grid(row=2,
                                                                                 column=0, padx=5,
                                                                                 pady=5, sticky=W)
positive_review = Scale(Input_frame, from_=0, to=100, resolution=1, orient=HORIZONTAL)
positive_review.grid(row=2, column=1, padx=5, pady=5, sticky=W)

Label(Input_frame, text='Number of Trailers', bg='grey').grid(row=0, column=2,
                                                              padx=5, pady=5, sticky=W)
num_of_movies = Entry(Input_frame, width=5)
num_of_movies.grid(row=0, column=3, padx=5, pady=5, sticky=W)

Label(Input_frame, text='Number Of Screenshot', bg='grey').grid(row=1, column=2, padx=5, pady=5,
                                                                sticky=W)
num_of_screenshots = Entry(Input_frame, width=5)
num_of_screenshots.grid(row=1, column=3, padx=5, pady=5, sticky=W)

Label(Input_frame, text='Supported Languages', bg='grey').grid(row=2, column=2, padx=5, pady=5,
                                                               sticky=W)

languagesList = []

regex = re.compile(r'[\n]')
with open('languagesList.txt', 'r') as inputLanguageFile:
    for language in inputLanguageFile:
        languageConvert = regex.sub("", language)
        languagesList.append(languageConvert)

inputLanguageFile.close()

languagesVarList = {}
language_regex = re.compile(r'[ -]')
count_row = 0
count_column = 0
for index in range(len(languagesList)):
    languageCompatible = language_regex.sub('', languagesList[index])
    languagesVarList[languageCompatible] = BooleanVar()
    languageVar = Checkbutton(Input_frame, text=languagesList[index], variable=languagesVarList[languageCompatible],
                              onvalue=True, offvalue=False,
                              bg='red')
    if index % 4 == 0:
        languageVar.grid(row=2+count_row, column=3, padx=0, pady=0, sticky=W)
    elif index % 4 == 1:
        languageVar.grid(row=2+count_row, column=4, padx=0, pady=0, sticky=W)
    elif index % 4 == 2:
        languageVar.grid(row=2 + count_row, column=5, padx=0, pady=0, sticky=W)
    elif index % 4 == 3:
        languageVar.grid(row=2 + count_row, column=6, padx=0, pady=0, sticky=W)
        count_row += 1

genresList = ["RPG", "Action", "Adventure", "Casual",
              "Indie", "Massively Multiplayer", "Racing",
              "Simulation", "Sports", "Strategy"]

Label(Input_frame, text='Genres', bg='grey').grid(row=3, column=0, padx=5, pady=5, sticky=W)
genresVarList = {}
genres_regex = re.compile(r'[ ]')
count_row = 0
for index in range(len(genresList)):
    genreCompatible = genres_regex.sub('', genresList[index])
    genresVarList[genreCompatible] = BooleanVar()
    genreVar = Checkbutton(Input_frame, text=genresList[index],
                           variable=genresVarList[genreCompatible], onvalue=True,
                           offvalue=False, bg='green')
    if index % 2 == 0:
        genreVar.grid(row=3+count_row, column=1, padx=0, pady=0,
                      sticky=W)
    else:
        genreVar.grid(row=3+count_row, column=2, padx=0, pady=0,
                      sticky=W)
        count_row += 1

description_row = 3+count_row+1

Label(Input_frame, text='Description', bg='grey').grid(row=description_row, column=0, padx=5,
                                                       pady=5)
description = Text(Input_frame, width=50, height=10)
description.grid(row=description_row, column=1, padx=5, pady=5)

Button(Input_frame, text='Predict', command=predict, bg='white').grid(row=0, column=5,
                                                                      padx=5, pady=5)
Label(Input_frame, text='Developer Name(sep by comma)', bg='grey').grid(row=1, column=5, padx=5,
                                                                        pady=5, sticky=W)
developers = Entry(Input_frame, width=10)
developers.grid(row=1, column=6, padx=5, pady=5, sticky=W)
root.mainloop()
