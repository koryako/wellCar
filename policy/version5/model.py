import data_prep
import model_creation
import json
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.python.control_flow_ops = tf


x_train, x_validation, y_train, y_validation = data_prep.prep_data()

datagen = ImageDataGenerator()
model = model_creation.get_model()
model.compile('adam', 'mean_squared_error', ['accuracy'])
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), samples_per_epoch=len(x_train),
                              nb_epoch=5, validation_data=datagen.flow(x_validation, y_validation, batch_size=32),
                              nb_val_samples=len(x_validation))
print(history)

model.save_weights("model.h5", True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
