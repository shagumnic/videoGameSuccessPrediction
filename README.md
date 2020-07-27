# videoGameSuccessPrediction
A video game success prediction application (predict number of players after a month) using machine learning and neural network:
- model_building: building prediction model based on the data downloaded from the Games-Data-Scraping-Download project (data was saved in dataForMachineLearning.csv):
	+ build_prediction_model_new.py: load the data matrix using pandas library from the csv file. The column will be used is: original_price (game's launch price), genres, positive_review (critics positive review percent), developers, num_of_movies (number of game's trailers), num_of_screenshots (number of game's screenshots), game_description_len (length of the game's description), languages (supported languages), num_players_after_month (number of players after a month), description. Process and feed the each column into the model as appropriate feature (numerical features will be 'positive_review', 'num_of_movies', 'num_of_screenshots', 'game_description_len', 'num_players_after_month', 'genres' (after one-hot encoding each genre), 'languages' (after one-hot encoding each supported language), 'num_of_languages' (after counting how many languages would be supported)) (bucket numerical column will be 'original_price') (embedding columns will be description and developers). Inspect the data using seaborn to plot it. Split the data into train, validation and test set (3:1:1 ratio). Normalized the data using Z Normalization formula. Create the model with all those features as input layers, 2 hidden layers with L2 and dropout regularization to prevent overfiting. Calculate the loss using mean squared error, early stopped if after 10 epochs, the loss doesn't decrease. Train the model with preset learning rate, epochs, batch size. Afterward, plot the loss using matplotlib. Then, using test set, compare the labels(actual result) with predicted one and plot the differences using matplotlib. Finally, save the model to the my_model folder for predicting. If wanted, users could customize the learning rate, epochs, the batch size to improve the model if they wanted.
	+ build_prediction_model_old.py: old version, use less features and only linear regression instead of neural network.
	+ languagesList.txt: complete list of languages a game could supported.
- predicting: build a GUI for the application:
	+ predicting.py: load the pre-built model, using tkinter to create the GUI for the application. Let the user to type in necessary information about their game for the prediction (name, developers, genres, supported languages, critics positive review percentage, number of game's trailers and screenshots, description of the game) and use the loaded model to predict the numbers of players after a month and pop up the result on the screen for the user.
	+ predicting_old.py: old version for the build_prediction_model_old.py
	+ languagesList.txt: complete list of languages a game could supported.
- predicting_gui: pyinstaller has convert the predicting.py file to an executable file. Because of the compatibility between pyinstaller and tensorflow, the whole folder couldn't be contain in a single .exe file. Therefore, it could only be convert into a folder. User just need to run the .exe file.

# How to use:
- navigate to predicting_gui folder and run the predicting.exe file
- if users want to customize and create their own model, install the required packages in the requirements.txt file and use the build_prediction_model_new.py to create their own model saved in my_model folder. Afterwards, drag the folder to the predicting_gui folder and run the predicting.exe file again.

# Tool use:
- Python
- tkinter to create GUI for the application
- pandas library to load the data .csv file
- numpy library to convert the data into matrix array for tensorflow to work with
- seaborn and matplotlib to plot the data.
- tensorflow (specifically tensorflow.keras) library for machine learning and neural network (building predicting model and train it to make future prediction).
- rake-nltk library to get the key words from game's description.
- re library to use regex to remove special characters from the the data.
- pyinstaller to convert the application into an .exe file
