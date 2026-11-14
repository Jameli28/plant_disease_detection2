import gradio as gr
import tensorflow as tf
import numpy as np

# Load your saved model and disease names
model = tf.keras.models.load_model("plant_model.keras")
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_disease(img):
    if img is None:
        return {}
    img = img.convert("RGB")  # Ensure image has 3 color channels
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
   
    predictions = model.predict(img_array)[0]  # Extract first row of predictions
    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

# Create simple browser UI
app = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=3, label="Detected Disease"),
    title="MCA Project: Plant Disease Detection"
)

app.launch()