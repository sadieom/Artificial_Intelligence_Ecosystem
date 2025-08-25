import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')

# Load model with imagenet weights
base_model = MobileNetV2(weights="imagenet")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the heatmap to match the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    # Apply the colormap
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    # Superimpose the heatmap on original image
    overlay = heatmap_color * alpha + img
    overlay = np.uint8(overlay)
    return overlay

def classify_and_gradcam(image_path, top=3):
    # Preprocess input image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Get model predictions
    preds = base_model.predict(img_array)
    decoded = decode_predictions(preds, top=top)[0]

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, base_model, 'Conv_1')
    overlay = overlay_heatmap(image_path, heatmap)

    # Save the overlay image
    out_path = image_path.rsplit('.', 1)[0] + '_gradcam.jpg'
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB_BGR))

    print(f"Top-{top} Predictions for {image_path}:")
    for i, (_, label, score) in enumerate(decoded):
        print(f"  {i+1}: {label} ({score:.2f})")
    print(f"Grad-CAM overlay saved to: {out_path}")

if __name__ == "__main__":
    print("Grad-CAM Image Classifier (type 'exit' to quit)\n")
    while True:
        path = input("Enter image filename: ").strip()
        if path.lower() == 'exit':
            print('Goodbye!')
            break
        classify_and_gradcam(path)