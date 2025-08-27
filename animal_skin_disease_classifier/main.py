import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configs
img_size = 224
batch_size = 32

# Manual class mapping
class_mapping = {
    "cat_Flea_Allergy": 0,
    "cat_Health": 1,
    "cat_Ringworm": 2,
    "cat_Scabies": 3,
    "dog_Bacterial_dermatosis": 4,
    "dog_Fungal_infections": 5,
    "dog_Healthy": 6,
    "dog_Hypersensitivity_allergic_dermatosis": 7
}
class_names = list(class_mapping.keys())

def load_data_from_folders(base_dir, img_size=224):
    images, labels = [], []
    for cls in class_names:
        class_path = os.path.join(base_dir, cls)
        if not os.path.exists(class_path):
            continue
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(class_mapping[cls])

    return np.array(images), np.array(labels), class_mapping

def get_feature_model(base_model, preprocess):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
        tf.keras.layers.Lambda(preprocess),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    return model

def build_classifier(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def extract_features(model, images):
    return model.predict(images, batch_size=batch_size, verbose=1)


# ------------------- FLASK API SECTION -------------------
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import io
from PIL import Image

app = Flask(__name__)

# Load models and classifier for inference
resnet = get_feature_model(ResNet50(weights='imagenet', include_top=False), resnet_preprocess)
effnet = get_feature_model(EfficientNetB0(weights='imagenet', include_top=False), efficientnet_preprocess)
densenet = get_feature_model(DenseNet121(weights='imagenet', include_top=False), densenet_preprocess)
for model in [resnet, effnet, densenet]:
    model.trainable = False

classifier = None
try:
    classifier = load_model("animal_skin_model_combined.h5")
except Exception as e:
    print("Warning: Could not load trained classifier. Prediction endpoint will not work until model is trained.")

def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((img_size, img_size))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if classifier is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files['file']
    img_bytes = file.read()
    img = prepare_image(img_bytes)
    # Extract features from all three models
    res_feat = resnet.predict(img)
    eff_feat = effnet.predict(img)
    den_feat = densenet.predict(img)
    feats = np.concatenate([res_feat, eff_feat, den_feat], axis=1)
    pred = classifier.predict(feats)
    pred_class = int(np.argmax(pred))
    pred_label = class_names[pred_class]
    return jsonify({"predicted_class": pred_label, "probabilities": pred[0].tolist()})

@app.route("/class_mapping", methods=["GET"])
def get_class_mapping():
    return jsonify(class_mapping)

if __name__ == "__main__":
    # Uncomment below to train the model
    # base_dir = "combined_dataset"
    # ...existing code for training...
    # print("Model training complete. Final model saved as animal_skin_model_combined.h5")
    app.run(host="0.0.0.0", port=5000, debug=True)