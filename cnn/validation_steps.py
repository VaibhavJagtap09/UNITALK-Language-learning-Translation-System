from tensorflow.keras.preprocessing.image import ImageDataGenerator

def calculate_validation_steps(directory, target_size, batch_size):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_set = test_datagen.flow_from_directory(directory,
                                                target_size=target_size,
                                                batch_size=batch_size,
                                                color_mode='grayscale',
                                                class_mode='categorical',
                                                shuffle=False)

    total_test_samples = len(test_set.filenames)
    validation_steps = total_test_samples // test_set.batch_size

    return validation_steps

# Example usage:
if __name__ == "__main__":
    directory = 'processed_images/test'  # Replace with your test dataset directory
    target_size = (128, 128)  # Replace with your target image size
    batch_size = 10  # Replace with your batch size for validation

    validation_steps = calculate_validation_steps(directory, target_size, batch_size)
    print("Validation Steps:", validation_steps)
