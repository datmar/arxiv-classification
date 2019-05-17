import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import re
from pathlib import Path 

import tensorflow as tf
import tensorflow.keras.preprocessing.text as Text
import tensorflow.keras.preprocessing.sequence as Sequence

df = pd.read_pickle('./data/two-topic-fulltext.pkl')

def clean_text(text):
    """
    Set text to lowercase and replace all non-ascii characters with spaces.
    """
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    return text

df["full_text"] = df["full_text"].apply(lambda x: clean_text(x))

# If tokenized text sequences exist, load as X, else tokenize the dataset and save.
my_file = Path("./data/text-sequences.npy")

if my_file.is_file():
    X = np.load("./data/text-sequences.npy")
else:
    tokenizer = Text.Tokenizer(num_words=20000,
                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                               lower=True, split=' ', char_level=False,
                               oov_token=None)
    tokenizer.fit_on_texts(df["full_text"].values)

    index_words = tokenizer.word_index
    print(f"Number of words: {len(index_words)}")

    X = tokenizer.texts_to_sequences(df["full_text"].values)

    X = Sequence.pad_sequences(X, maxlen=250)

    np.save("./data/text-sequences.npy", X)

Y = df["topic"].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=123)

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(monitor='val_acc', min_delta=.0001, patience=4, verbose=True)]

model = Sequential()
model.add(layers.Embedding(20_000, 100, input_length=X.shape[1]))
model.add(layers.LSTM(100, dropout=0.25, recurrent_dropout=0.25))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=callbacks)

model.evaluate(X_test,Y_test)

model.save("./results/paper_classifier2.h5")

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-deep')

plt.figure(figsize=(5, 4))
plt.title("Accuracy")
plt.plot(history.history["accuracy"], label='train')
plt.plot(history.history["val_accuracy"], label='test')
plt.legend()
plt.show()
plt.savefig("./results/accuracy-text-classifier.png")

plt.figure(figsize=(5, 4))
plt.title("Loss")
plt.plot(history.history["loss"], label='train')
plt.plot(history.history["val_loss"], label='test')
plt.legend()
plt.show()
plt.savefig("./results/loss-text-classifier.png")

Y_test_predict = model.predict(X_test)

plt.figure(figsize=(5, 4))
plt.hist(Y_test_predict[Y_test == 0], alpha = 0.5)
plt.hist(Y_test_predict[Y_test == 1], alpha = 0.5)
plt.legend(["Deep Learning", "Computer Vision"])
plt.title("Prediction Confidence for Both Topics")
plt.show()
plt.savefig("./results/prediction-confidences.png")

uncertain_predictions_mask = (Y_test_predict > 0.4) & (Y_test_predict < 0.6)
X_uncertain = X_test[uncertain_predictions_mask.flatten()]

df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df["full_text"].values, df["topic"].values,
                                                                test_size=0.15, random_state=123)

uncertain_papers = df_x_test[uncertain_predictions_mask.flatten()]
print("Full text of uncertain prediction:")
uncertain_papers[1]

test_loss = model.evaluate(X_test, Y_test)

