from get_data1 import training_images, training_labels,testing_images, testing_labels
from import_files import *

training_images = np.expand_dims(training_images,axis=3)
testing_images = np.expand_dims(testing_images,axis=3)


train_datagen = ImageDataGenerator(rescale=1.0/255,
   rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

validation_datagen = ImageDataGenerator(
    rescale=1.0/255)
    

print(training_images.shape)
print(testing_images.shape)