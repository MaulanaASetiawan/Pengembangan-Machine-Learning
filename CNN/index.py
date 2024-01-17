# %% [markdown]
# ### **Nama : Maulana Agus Setiawan**
# ### **Link Dataset : https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset**

# %% [markdown]
# #### Import Library

# %%
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

# %% [markdown]
# #### Load Data

# %%
path = 'PetImages'
files = os.listdir(path)

for file in files:
    print(file)

# %% [markdown]
# #### Remove file yang berekstensi Zone.Identifier

# %%
import os

def delete_identifier_files(folder_path):
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith(":Zone.Identifier"):
                file_path = os.path.join(root, file_name)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Gagal menghapus file {file_path}. Error: {e}")

cat = os.path.join(path, '/Cat')
dog = os.path.join(path, '/Dog')

delete_identifier_files(cat)
delete_identifier_files(dog)

# %% [markdown]
# #### Display Gambar

# %%
def display_image(path, category):
    image_files = os.listdir(os.path.join(path, category))
    selected_images = image_files[:3]
    plt.figure(figsize=(15, 7))
    for i, image_name in enumerate(selected_images, 1):
        image_path = os.path.join(path, category, image_name)
        image = Image.open(image_path)
        plt.subplot(2, 5, i)
        plt.imshow(image)
        plt.title(f"Gambar {i}")
        plt.axis("off")
    plt.suptitle(category.capitalize(), fontsize=15)
    plt.show()

for file in files:
    display_image(path, file)

# %% [markdown]
# #### Data Info

# %%
jumlah_data = 0
for animal in files:
    images = len(os.listdir(f'{path}/{animal}'))
    print(f'{animal} images: ', images)
    jumlah_data += images
print (f'Total Images : {jumlah_data}')

# %% [markdown]
# #### Split Data dengan rasio 80:20

# %%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# %% [markdown]
# 

# %%
train_generator = train_datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size=1,
    class_mode='binary',
    subset='validation'
)

# %%
class_names = list(train_generator.class_indices.keys())
class_names

# %% [markdown]
# #### Modelling Sequential

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# %%
model.summary()

# %%
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.optimizers.Adam(),
    metrics=['accuracy'],
)

# %% [markdown]
# #### Fungsi Callbacks

# %%
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.93 and logs.get('val_accuracy') > 0.93):
            self.model.stop_training = True
            print("\nAkurasi telah terpenuhi > 93%!")

callbacks = myCallback()

# %% [markdown]
# #### Training Model

# %%
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    verbose=1,
    callbacks=[callbacks]
)

# %% [markdown]
# #### Evaluasi Model

# %%
model.evaluate(val_generator)

# %% [markdown]
# #### Grafik pergerakan metrik akurasi dan loss pada training maupun validation

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

# %% [markdown]
# #### Menyimpan Model dalam format TF-Lite

# %%
save = 'saved_model'
tf.saved_model.save(model, save)

# %%
converter = tf.lite.TFLiteConverter.from_saved_model(save)
tflite_model = converter.convert()

with open('MyModel.tflite', 'wb') as f:
    f.write(tflite_model)


