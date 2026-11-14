import matplotlib.pyplot as plt

# आपके असली ट्रेनिंग रिजल्ट्स (Epochs 1 से 5 तक)
epochs = [1, 2, 3, 4, 5]
train_acc = [86.47, 97.03, 97.87, 98.60, 98.84]
val_acc = [95.68, 96.54, 96.86, 97.19, 97.19]

train_loss = [0.4542, 0.1318, 0.0917, 0.0697, 0.0566]
val_loss = [0.1926, 0.1475, 0.1050, 0.0942, 0.0885]

# ग्राफ की स्टाइल सेट करें
plt.figure(figsize=(12, 5), dpi=300)

# 1. Accuracy Graph (बायां हिस्सा)
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo-', linewidth=2, markersize=6, label='Training Accuracy')
plt.plot(epochs, val_acc, 'go-', linewidth=2, markersize=6, label='Validation Accuracy')
plt.title('Model Accuracy (Training vs Validation)', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.xticks(epochs)
plt.ylim(80, 100)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')

# 2. Loss Graph (दायां हिस्सा)
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'ro-', linewidth=2, markersize=6, label='Training Loss')
plt.plot(epochs, val_loss, 'mo-', linewidth=2, markersize=6, label='Validation Loss')
plt.title('Model Loss (Training vs Validation)', fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=11)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

# इमेज को हाई-क्वालिटी PNG में सेव करें
plt.tight_layout()
plt.savefig('training_results_graph.png', dpi=300)
print("SUCCESS! 'training_results_graph.png' आपके फोल्डर में सेव हो गई है।")
plt.show()