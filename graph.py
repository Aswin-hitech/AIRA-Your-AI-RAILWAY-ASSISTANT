import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import make_interp_spline

# ------------------------------
# Example Epoch-wise Data (Distinct & Realistic)
# ------------------------------
epochs = np.arange(1, 11)

training_accuracy = [0.78, 0.82, 0.85, 0.87, 0.89, 0.91, 0.925, 0.93, 0.935, 0.94]
validation_accuracy = [0.75, 0.80, 0.83, 0.86, 0.885, 0.90, 0.915, 0.925, 0.93, 0.935]
training_loss = [0.82, 0.71, 0.60, 0.51, 0.42, 0.36, 0.32, 0.29, 0.26, 0.24]
validation_loss = [0.84, 0.74, 0.63, 0.54, 0.45, 0.38, 0.34, 0.30, 0.27, 0.25]

# Create output directory
os.makedirs("metrics", exist_ok=True)

# ------------------------------
# Helper function for smooth curve plotting
# ------------------------------
def smooth_curve(x, y, points=300):
    x_smooth = np.linspace(x.min(), x.max(), points)
    y_smooth = make_interp_spline(x, y, k=3)(x_smooth)
    return x_smooth, y_smooth

# ------------------------------
# 1) Training and Validation (Accuracy only)
# ------------------------------
plt.figure(figsize=(8,5))
x_smooth, y_train_smooth = smooth_curve(epochs, training_accuracy)
_, y_val_smooth = smooth_curve(epochs, validation_accuracy)

plt.plot(x_smooth, y_train_smooth, label="Training Accuracy", color='blue', linewidth=2)
plt.plot(x_smooth, y_val_smooth, label="Validation Accuracy", color='orange', linewidth=2)
plt.scatter(epochs, training_accuracy, color='blue', s=50)
plt.scatter(epochs, validation_accuracy, color='orange', s=50)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy vs Epochs")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("metrics/training_validation_accuracy.png", dpi=300)
plt.show()

# ------------------------------
# 2) Accuracy Model Graph (Accuracy vs Epochs)
# ------------------------------
plt.figure(figsize=(8,5))
x_smooth, y_train_smooth = smooth_curve(epochs, training_accuracy)
_, y_val_smooth = smooth_curve(epochs, validation_accuracy)

plt.plot(x_smooth, y_train_smooth, label="Training Accuracy", color='green', linewidth=2)
plt.plot(x_smooth, y_val_smooth, label="Validation Accuracy", color='red', linewidth=2)
plt.scatter(epochs, training_accuracy, color='green', s=50)
plt.scatter(epochs, validation_accuracy, color='red', s=50)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Epochs")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("metrics/model_accuracy_curve.png", dpi=300)
plt.show()

# ------------------------------
# 3) Model Loss vs Epochs (Training & Validation)
# ------------------------------
plt.figure(figsize=(8,5))
x_smooth, y_train_smooth = smooth_curve(epochs, training_loss)
_, y_val_smooth = smooth_curve(epochs, validation_loss)

plt.plot(x_smooth, y_train_smooth, label="Training Loss", color='navy', linewidth=2)
plt.plot(x_smooth, y_val_smooth, label="Validation Loss", color='crimson', linewidth=2)
plt.scatter(epochs, training_loss, color='navy', s=50)
plt.scatter(epochs, validation_loss, color='crimson', s=50)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss vs Epochs")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("metrics/model_loss_curve.png", dpi=300)
plt.show()

# ------------------------------
# 4) Training and Validation (Loss only)
# ------------------------------
plt.figure(figsize=(8,5))
x_smooth, y_train_smooth = smooth_curve(epochs, training_loss)
_, y_val_smooth = smooth_curve(epochs, validation_loss)

plt.plot(x_smooth, y_train_smooth, label="Training Loss", color='teal', linewidth=2)
plt.plot(x_smooth, y_val_smooth, label="Validation Loss", color='darkorange', linewidth=2)
plt.scatter(epochs, training_loss, color='teal', s=50)
plt.scatter(epochs, validation_loss, color='darkorange', s=50)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss vs Epochs")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("metrics/training_validation_loss.png", dpi=300)
plt.show()

print("✅ All four smooth curved graphs saved in 'metrics/' folder.")
