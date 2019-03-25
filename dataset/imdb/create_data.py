from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py


def load_original_data(data_type='train'):
    imgs = []
    labels = []
    img_datagen = ImageDataGenerator(rescale=1.0 / 255)
    if data_type == "train":
        img_generator = img_datagen.flow_from_directory(
            'dataset/agegender_imdb/annotations/gender/train',
            target_size=(64, 64),
            batch_size=1,
            class_mode='categorical',
            shuffle=True
        )

        for i in range(len(img_generator)):
            imgs.append(img_generator[i][0][0])
            labels.append(img_generator[i][1][0])
        #
        # total_images = len(img_generator)
        # train_shape = (total_images, 64, 64, 3)
        # imgs = np.array(imgs).reshape(train_shape).tolist()
        #
        # test_shape = (total_images, 2)
        # labels = np.array(labels).reshape(test_shape)
        return np.array(imgs), np.array(labels)


def load_original_test_data():
    imgs = []
    labels = []
    img_datagen = ImageDataGenerator(rescale=1.0 / 255)
    img_generator = img_datagen.flow_from_directory(
        'dataset/agegender_imdb/annotations/gender/validation',
        target_size=(64, 64),
        batch_size=1,
        class_mode='categorical',
        shuffle=True
    )
    for i in range(len(img_generator)):
        imgs.append(img_generator[i][0][0])
        labels.append(img_generator[i][1][0])

    return np.array(imgs), np.array(labels)


def write_file():
    hf = h5py.File("test.h5", 'w')
    imgs, labels = load_original_test_data()
    hf.create_dataset('X', data=imgs)
    hf.create_dataset('Y', data=labels)
    hf.close()

    hf = h5py.File("train.h5", 'w')
    imgs, labels = load_original_data()
    hf.create_dataset('X', data=imgs)
    hf.create_dataset('Y', data=labels)
    hf.close()


write_file()

