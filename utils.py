from keras.preprocessing import image


def get_batches(
        dirname,
        gen=image.ImageDataGenerator(),
        shuffle=False,
        save_to_dir=None,
        batch_size=32,
        class_mode='categorical',
        target_size=(128, 128)):
    return gen.flow_from_directory(
            dirname,
            # color_mode='rbg',
            save_to_dir=save_to_dir,
            target_size=target_size,
            class_mode=class_mode,
            shuffle=shuffle,
            batch_size=batch_size)
