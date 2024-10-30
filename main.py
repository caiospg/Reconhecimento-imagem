import tensorflow as tf
from tensorflow.keras import layers, models

# Carregar o dataset com divisão
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\AmbienteDesenvolvimento\ProjetoIA2.1\datasetOil\dataset',
    image_size=(224, 224),
    batch_size=16,
    label_mode='categorical',
    validation_split=0.2,  # 20% para validação
    subset="training",      # Para o conjunto de treinamento
    seed=123               # Para reprodutibilidade
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\AmbienteDesenvolvimento\ProjetoIA2.1\datasetOil\dataset',
    image_size=(224, 224),
    batch_size=16,
    label_mode='categorical',
    validation_split=0.2,  # 20% para validação
    subset="validation",    # Para o conjunto de validação
    seed=123               # Para reprodutibilidade
)

# Se você tiver um conjunto de teste separado, carregue-o assim:
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\AmbienteDesenvolvimento\ProjetoIA2.1\datasetOil\dataset/test',
    image_size=(224, 224),
    batch_size=16,
    label_mode='categorical'
)

# Definir um modelo básico (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 2 classes para softmax
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo usando o conjunto de treinamento e validação
model.fit(train_dataset, validation_data=validation_dataset, epochs=6)