import tensorflow as tf
from tensorflow.keras import layers, models

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_size=(224, 224),
    batch_size=16,
    label_mode='categorical',
    subset="training",      # Para o conjunto de treinamento
    seed=123               # Para reprodutibilidade
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_size=(224, 224),
    batch_size=16,
    label_mode='categorical',
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_size=(224, 224),
    batch_size=16,
)

# Definir um modelo básico (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo usando o conjunto de treinamento e validação