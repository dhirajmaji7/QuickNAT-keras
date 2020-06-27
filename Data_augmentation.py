from keras.preprocessing.image import ImageDataGenerator

image_datagen = ImageDataGenerator(rotation_range=10,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1)
mask_datagen = ImageDataGenerator(rotation_range=10,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1)


# X_train should contain the training images and Y_train should contain the corresponding ground truths.
# Using the same seed value will provide same augmentations to both the training and ground truth images.

seed = 1
image_generator = image_datagen.flow(X_train, batch_size=12, seed=seed)
mask_generator = mask_datagen.flow(Y_train, batch_size=12, seed=seed)
train_generator = zip(image_generator, mask_generator)

# During training, use fit_generator function as show below : 
# model.fit_generator(train_generator,
#	                  validation_data=(X_valid, Y_valid),
#	                  steps_per_epoch=len(X_train) // batch_size,
#	                  epochs=epochs, verbose=1, shuffle=True)

