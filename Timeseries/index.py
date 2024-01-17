# %% [markdown]
# ### **Nama : Maulana Agus Setiawan**
# ### **Dataset : https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption**

# %% [markdown]
# #### **Import Library**

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# %% [markdown]
# #### **Read Dataset**

# %%
df = pd.read_csv('Steel_industry_data.csv')
df.tail()

# %%
df.info()

# %% [markdown]
# #### **Mengecek Nilai Null**

# %%
df.isna().sum()

# %% [markdown]
# #### **Normalisasi Data**

# %%
mm = MinMaxScaler()
df['Usage_kWh'] = mm.fit_transform(df[['Usage_kWh']])

# %% [markdown]
# #### **Splitting Data**

# %%
date = df['date'].values
kwh = df['Usage_kWh'].values

X_train, X_test = train_test_split(kwh, test_size=0.2, shuffle=False)

# %%
print(X_train.shape)
print(X_test.shape)

# %%
plt.figure(figsize=(15,8))
plt.plot(date, kwh)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Usage kWh', fontsize=20)
plt.title('Usage kWh Average', fontsize=20);

# %% [markdown]
# #### **Modelling Sequential**

# %%
# Merubah Data untuk dapat diterima model
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)

# %%
# Modelling data
train_set = windowed_dataset(
    X_train,
    window_size=60,
    batch_size=100,
    shuffle_buffer=1000
)

val_set = windowed_dataset(
    X_test,
    window_size=60,
    batch_size=100,
    shuffle_buffer=1000
)

model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(60, return_sequences=True),
  tf.keras.layers.LSTM(60),
  tf.keras.layers.Dense(64, activation="relu"),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(16, activation="relu"),
  tf.keras.layers.Dense(1),
])

# %%
threshold_mae = (df['Usage_kWh'].max() - df['Usage_kWh'].min()) * 10/100
print(threshold_mae)

# %%
class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      if(logs.get('val_mae') < threshold_mae and logs.get('mae') < threshold_mae):
        print(f"\nMAE telah berada dibawah : {threshold_mae}")
        self.model.stop_training = True

callbacks = MyCallBack()

# %%
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1.0000e-04
)

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"]
)

history = model.fit(
    train_set,
    epochs=100,
    validation_data = val_set,
    callbacks = [callbacks]
)

# %%
predictions = model.predict(X_test[-60:][np.newaxis])
print(predictions)

# %%
print("Mean Absolute Error (MAE):", predictions)
print("Threshold MAE:", threshold_mae)

if predictions <= threshold_mae:
    print("-> BAIK")
else:
    print("-> TIDAK BAIK")


