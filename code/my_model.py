# Run in google colab

# Importing necessary libraries such as pandas, re, numpy, keras, scikit-learn, and matplotlib
import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Reading the dataset from a CSV file
path = "/content/gdrive/MyDrive/RIT/SEM2/Software Engineering for Data science/lingustic_quality_inter.csv"
data = pd.read_csv(path)

# Defining a function called clean_column to remove special characters from a column 
# and convert all data to lowercase - data preprocessing
def clean_column(col):
    col = col.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]','',x))
    col = col.str.lower()
    return col

# Applying the clean_column function to the specified columns in the dataset
cols_to_clean = ['log_messages', 'log_level', 'lemmas', 'pos', 'tag', 'dep']
for col in cols_to_clean:
    data[col] = clean_column(data[col])

# Extracting the categorical columns from the dataset and store them in a separate variable
categorical_cols = data[['log_messages', 'log_level', 'lemmas', 'pos', 'tag', 'dep']]

# Apply one-hot encoding to the categorical columns to convert them into numerical values
enc = OneHotEncoder()
enc.fit(categorical_cols)
encoded_data = enc.transform(categorical_cols).toarray()

#  Extract the numerical columns from the dataset and store them in a separate variable
numerical_data = data.drop(columns=['log_messages', 'log_level', 'lemmas', 'pos', 'tag', 'dep', 'label'])
y = data['label']
# Combine the one-hot encoded categorical columns and numerical columns.
X = np.hstack((encoded_data, numerical_data))

# Split the dataset into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Using feature selection to select the top 10 most relevant features using the F-test and 
# select the corresponding columns from the training, validation, and testing sets
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

# Creating a sequential neural network model using Keras with 
# 16 hidden units and a sigmoid activation function for the output layer
model = Sequential()
model.add(Dense(16, input_dim=X_train_selected.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model with binary cross-entropy loss and an RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.5)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Training the model on the training set for 5 epochs and a batch size of 32, 
# and validate the performance on the validation set
model.fit(X_train_selected, y_train, epochs=5, batch_size=32, validation_data=(X_val_selected, y_val))

# Evaluating the model on the testing set and print the test accuracy
test_loss, test_acc = model.evaluate(X_test_selected, y_test)
print('Test accuracy:', test_acc)

# Saving the model to Google Drive in the specified location
model.save('/content/gdrive/MyDrive/RIT/SEM2/Software Engineering for Data science/model/my_model.h5')

# Using the trained model to predict the labels of the testing set
y_pred = np.round(model.predict(X_test)).astype(int)

# Calculating the precision score and recall on the testing set
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(precision)
print(recall)

# Creating a confusion matrix and plot it using Matplotlib
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
