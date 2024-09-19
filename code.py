import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import argparse
import sys

# Set random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Constants
IMAGE_SIZE = (224, 224)  # MobileNetV2 expects 224x224 images
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

def prepare_data(train_dir, validation_dir):
    """
    Prepare training and validation data generators.
    """
    # Define ImageDataGenerator for training with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # ImageDataGenerator for validation (only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED
    )

    # Flow validation images in batches
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED
    )

    return train_generator, validation_generator

def build_model():
    """
    Build the CNN model using MobileNetV2 as the base.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the base model

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # Combine base model and custom layers
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

def train_model(model, train_generator, validation_generator, model_save_path='fake_news_model.h5'):
    """
    Train the model with early stopping and save the best model.
    """
    # Define callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stop, checkpoint],
        verbose=2
    )

    return history

def save_model_artifacts(model, model_path='fake_news_model.h5'):
    """
    Save the trained model to disk.
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_trained_model(model_path='fake_news_model.h5'):
    """
    Load a trained model from disk.
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        sys.exit(1)
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

def preprocess_image(image_path):
    """
    Load and preprocess a single image for prediction.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def predict_image(model, image_path):
    """
    Predict whether the given image is fake or real news.
    """
    img = preprocess_image(image_path)
    if img is None:
        return
    prediction = model.predict(img)[0][0]
    label = 'Fake' if prediction >= 0.5 else 'Real'
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    print(f"Image: {os.path.basename(image_path)} | Prediction: {label} (Confidence: {confidence:.4f})")

def predict_from_directory(model, images_dir):
    """
    Predict fake or real for all images in a given directory.
    """
    if not os.path.exists(images_dir):
        print(f"Images directory {images_dir} does not exist.")
        return

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"No supported image files found in {images_dir}.")
        return

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        predict_image(model, image_path)

def main():
    parser = argparse.ArgumentParser(description='Image-Only Fake News Detection using Deep Learning')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Sub-parser for training
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--train_dir', type=str, required=True, help='Path to training images directory')
    train_parser.add_argument('--validation_dir', type=str, required=True, help='Path to validation images directory')
    train_parser.add_argument('--model_save_path', type=str, default='fake_news_model.h5', help='Path to save the trained model')

    # Sub-parser for prediction
    predict_parser = subparsers.add_parser('predict', help='Predict fake or real news from images')
    predict_parser.add_argument('--model_path', type=str, default='fake_news_model.h5', help='Path to the trained model')
    group = predict_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str, help='Path to a single image file for prediction')
    group.add_argument('--images_dir', type=str, help='Path to a directory containing images for prediction')

    args = parser.parse_args()

    if args.command == 'train':
        # Prepare data
        print("Preparing data...")
        train_gen, val_gen = prepare_data(args.train_dir, args.validation_dir)

        # Build model
        print("Building model...")
        model = build_model()

        # Train model
        print("Training model...")
        history = train_model(model, train_gen, val_gen, args.model_save_path)

        print("Training completed and model saved.")

    elif args.command == 'predict':
        # Load model
        model = load_trained_model(args.model_path)

        # Predict
        if args.image_path:
            print(f"Predicting for image: {args.image_path}")
            predict_image(model, args.image_path)
        elif args.images_dir:
            print(f"Predicting for all images in directory: {args.images_dir}")
            predict_from_directory(model, args.images_dir)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
