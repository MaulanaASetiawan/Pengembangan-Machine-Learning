# %% [markdown]
# ## **NLP SUBMISSION DICODING**
# ##### Nama    : Maulana Agus Setiawan
# ##### Dataset : https://www.kaggle.com/datasets/gregorut/videogamesales

# %% [markdown]
# <li>Import Library</li>

# %%
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# %% [markdown]
# <li>Read Data</li>

# %%
df = pd.read_csv('vgsales.csv')
df = df[['Name', 'Genre']]
df

# %%
df.info()

# %%
df.head()

# %% [markdown]
# <li>Encoding</li>

# %%
category = pd.get_dummies(df.Genre)
new_df = pd.concat([df, category], axis=1)
new_df = new_df.drop(columns='Genre')
new_df

# %% [markdown]
# <li>Cek Nilai Null</li>

# %%
new_df.isna().sum()

# %% [markdown]
# <li>Split Data</li>

# %%
X = new_df['Name'].values
y = new_df[[
    'Action', 'Adventure', 'Fighting', 'Misc', 
    'Platform', 'Puzzle', 'Racing', 'Role-Playing', 
    'Shooter', 'Simulation', 'Sports', 'Strategy'
]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% [markdown]
# <li>Tokenizing</li>

# %%
NW = 5000
oov_tok = '-'

tokenizer = Tokenizer(num_words=NW, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_test)

Xtrain_Seq = tokenizer.texts_to_sequences(X_train)
Xtest_Seq  = tokenizer.texts_to_sequences(X_test)

X_padded_train = pad_sequences(Xtrain_Seq)
X_padded_test = pad_sequences(Xtest_Seq)

# %% [markdown]
# <li>Modelling</li>

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=NW, output_dim=64),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(12, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

# %% [markdown]
# <li>Callbacks</li>

# %%
callbacks = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

# %% [markdown]
# <li>Proses Training</li>

# %%
epoch = 30
history = model.fit(
    X_padded_train,
    y_train,
    epochs=epoch,
    validation_data=(X_padded_test, y_test),
    verbose=1,
    callbacks=[callbacks]
)

# %% [markdown]
# <li>Evaluasi Model</li>

# %%
los, acc = model.evaluate(X_padded_test, y_test)

print("Loss : ", los)
print("Accuracy : ", acc)

# %% [markdown]
# <li>Plotting</li>

# %%
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Plot')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Plot')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()