import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


# Обработка папок с тестовыми и тренировочными фото
def load_data(data_dir):
    def load_subset(subset_dir):
        images = []     # Фото
        labels = []     # Типы облаков
        class_names = os.listdir(subset_dir)
        
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(subset_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)
                    try:
                        image = Image.open(file_path)
                        image = image.convert('RGB')  # Преобразование изображения в формат RGB
                        image = image.resize((128, 128))  # Изменение размера изображения
                        images.append(np.array(image))
                        labels.append(label)
                    except Exception as e:
                        print(f"Ошибка при загрузке {file_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    train_images, train_labels = load_subset(train_dir)
    test_images, test_labels = load_subset(test_dir)
    
    return (train_images, train_labels), (test_images, test_labels)

data_dir = 'Howard-Cloud-X'   #Датасет
(train_images, train_labels), (test_images, test_labels) = load_data(data_dir)

# Создание модели нейросети
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(train_labels)), activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# От переобучения
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение модели
history = model.fit(
    train_images, train_labels,
    epochs=50,
    batch_size=32,
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping]
)

# Оценка модели
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Test Accuracy: {accuracy}')

# Визуализация результатов обучения
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch') 
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Предсказание на новом фото
def predict_image(image_path):
    # Загрузка и подготовка изображения
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))
    img_array = np.expand_dims(img_resized, axis=0)
    
    # Предсказание класса
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    # Получение имени класса
    class_names = os.listdir(os.path.join(data_dir, 'train'))
    predicted_class_name = class_names[predicted_class]
    
    # Отображение
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Predicted Class: {predicted_class_name}')
    plt.axis('off')
    plt.show()
    
    return predicted_class_name

# Наше фото
test_image_path = 'check/cirr.jpg'                      # Вставть из папки "check"
predicted_cloud_type = predict_image(test_image_path)
print(f'The predicted cloud type is: {predicted_cloud_type}')
