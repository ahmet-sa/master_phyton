import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data_dir = r'C:\Users\ahmee\OneDrive\Masaüstü\YL\Pressure_Injury\dataset'

save_dir = './saved_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.35,
    height_shift_range=0.35,
    shear_range=0.35,
    zoom_range=0.35,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

def create_model(base_model):
    base_model.trainable = True  

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

base_models = {
    'DenseNet121': DenseNet121(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
}

epochs = 40
batch_size = 64

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=os.path.join(save_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True),
    LearningRateScheduler(scheduler)
]

for name, base_model in base_models.items():
    print(f"Training model: {name}")
    model = create_model(base_model)
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Validation Accuracy for {name}: {val_accuracy:.4f}")
    print(f"Validation Loss for {name}: {val_loss:.4f}\n")

    model_path = os.path.join(save_dir, f"{name}_pressure_injury_classifier.keras")
    model.save(model_path)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{name} - Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{name} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    
    plt.savefig(os.path.join(save_dir, f"{name}_training_plots.png"))
    plt.show()

    val_preds = model.predict(validation_generator)
    val_labels = validation_generator.classes
    class_names = validation_generator.class_indices.keys()
    
    y_pred = np.argmax(val_preds, axis=1)
    cm = confusion_matrix(val_labels, y_pred, labels=list(validation_generator.class_indices.values()))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'{name} - Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f"{name}_confusion_matrix.png"))
    plt.show()

def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    classes = ['Evre 1', 'Evre 2', 'Evre 3', 'Evre 4']
    print(f"Tahmin Edilen Sınıf: {classes[predicted_class]}")

# Example usage of the prediction function
model = tf.keras.models.load_model('./saved_models/DenseNet121_pressure_injury_classifier.keras')

# Predict using the model
# Replace 'path_to_image' with the path to the image you want to predict
