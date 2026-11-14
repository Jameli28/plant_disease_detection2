import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load dataset from the local folder
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset", validation_split=0.2, subset="training",
    seed=123, image_size=(224, 224), batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset", validation_split=0.2, subset="validation",
    seed=123, image_size=(224, 224), batch_size=32
)

class_names = train_ds.class_names
print("Diseases Found:", class_names)

# Save disease names for the app
with open("classes.txt", "w") as f:
    for item in class_names:
        f.write(f"{item}\n")

# 2. Build lightweight MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Train for 5 epochs
print("Starting training... please wait.")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 4. Save the trained model
model.save("plant_model.keras")
print("SUCCESS! Model saved as plant_model.keras")