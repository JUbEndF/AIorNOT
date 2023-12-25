from matplotlib import pyplot as plt
import cv2
import os
from fastai.vision.all import *
from fastbook import *
import os
import pywt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras import callbacks
from sklearn.metrics import roc_auc_score
import multiprocessing
folder      =  "D:\\Ai or not\\trein\\"     # "ImagesAI\\"
test_folder =  "D:\\Ai or not\\test\\"

def GaussianBlur(img):
    return img - cv2.GaussianBlur(img, (5, 5), 0)

def bilateralFilter(image):
    return image - cv2.bilateralFilter(image, 7, 65, 65)

def dwt2(image):
    coeffs2 = pywt.dwt2(image, 'bior1.3')
    threshold = 30
    coeffs2 = tuple(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs2))
    denoised_image = pywt.idwt2(coeffs2, 'bior1.3')
    denoised_image = denoised_image[:image.shape[0], :image.shape[1], :image.shape[2]]
    result = image - denoised_image
    return result


def medianBlur(image):
    return image - cv2.medianBlur(image, 5)

Epochs = 10

def trein_model(preprocessing_function):
    target_size=(200, 200)

    datagen = ImageDataGenerator(rescale=1./255,
                                validation_split=0.4,
                                preprocessing_function=preprocessing_function)

    train_generator = datagen.flow_from_directory(
        folder,
        target_size=target_size,  # Размер изображений
        batch_size=32,
        class_mode='categorical',  # для многоклассовой классификации
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        test_folder,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    opt = keras.optimizers.SGD(learning_rate=0.001)
    auc=tf.keras.metrics.AUC()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.AUC(name='auc')])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    history = model.fit(
        train_generator,
        epochs=Epochs,
        validation_data=validation_generator
    )
    model.save('my_model.keras')

    acc = history.history['auc']
    plt.plot(range(1, Epochs+1), acc, label=preprocessing_function.__name__)
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.savefig(preprocessing_function.__name__ + ".png")
    plt.close()
    return (preprocessing_function.__name__, acc)
            
if __name__ == '__main__':

    data = [medianBlur, GaussianBlur, bilateralFilter, dwt2]
    
    
    
    with multiprocessing.Pool() as pool:
        results = pool.map(trein_model,data)
    
    print(results)
    colors = ['red', 'blue', 'green', 'black','yellow', 'purple']
    my_iterator = iter(colors)
    for name, value in results:
        plt.plot(range(1, Epochs+1), value, label=name, color=next(my_iterator))

    plt.title('Зависимость AUC от функции удаления шума')
    plt.xlabel('epoch')    
    plt.ylabel('auc')
    plt.legend()
    plt.savefig("graf1.png")
    plt.show()