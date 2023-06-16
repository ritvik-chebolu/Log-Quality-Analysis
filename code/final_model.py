# Run in google colab

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


path = "/content/gdrive/MyDrive/RIT/SEM2/Software Engineering for Data science/lingustic_quality_inter.csv"
data = pd.read_csv(path)


def clean_column(col):
    col = col.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]','',x))
    col = col.str.lower()
    return col

cols_to_clean = ['log_messages', 'log_level', 'lemmas', 'pos', 'tag', 'dep']
for col in cols_to_clean:
    data[col] = clean_column(data[col])


categorical_cols = data[['log_messages', 'log_level', 'lemmas', 'pos', 'tag', 'dep']]

# one-hot encoding
# ct = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(), categorical_cols)],
#     remainder='passthrough'
# )
# encoded_data = ct.fit_transform(data)
# encoded_data.shape

enc = OneHotEncoder()
enc.fit(categorical_cols)
encoded_data = enc.transform(categorical_cols).toarray()

numerical_data = data.drop(columns=['log_messages', 'log_level', 'lemmas', 'pos', 'tag', 'dep', 'label'])
y = data['label']
X = np.hstack((encoded_data, numerical_data))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

model = Sequential()
model.add(Dense(16, input_dim=X_train_selected.shape[1], activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.RMSprop(learning_rate=0.5)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train_selected, y_train, epochs=5, batch_size=32, validation_data=(X_val_selected, y_val))

test_loss, test_acc = model.evaluate(X_test_selected, y_test)
print('Test accuracy:', test_acc)

model.save('/content/gdrive/MyDrive/RIT/SEM2/Software Engineering for Data science/model/my_model.h5')


y_pred = np.round(model.predict(X_test)).astype(int)


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(precision)
print(recall)


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()