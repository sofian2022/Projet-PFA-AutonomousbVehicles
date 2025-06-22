import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# === Chargement du modèle entraîné ===
model = load_model("../model/traffic_classifiernew.h5")

# === Dictionnaire des classes GTSRB ===
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
}

# Variables globales
image_files = []
current_index = 0
img_tk = None

# === Fonction de prédiction améliorée ===
def predict_image(file_path):
    image = cv2.imread(file_path)
    image_fromarray = Image.fromarray(image, 'RGB')
    resized_image = image_fromarray.resize((30, 30))
    image_array = np.array(resized_image) / 255.0
    image_array = image_array.reshape(1, 30, 30, 3)
    prediction = model.predict(image_array)
    confidence = np.max(prediction)
    class_index = np.argmax(prediction)
    
    if confidence < 0.8:
        return "No prediction (low confidence)", confidence
    else:
        return classes[class_index], confidence

# === Fonction pour afficher l'image actuelle ===
def show_current_image():
    global img_tk
    if 0 <= current_index < len(image_files):
        file_path = image_files[current_index]

        # Afficher l'image
        img = Image.open(file_path)
        img = ImageOps.fit(img, (300, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Obtenir la prédiction et la confiance
        class_name, confidence = predict_image(file_path)
        
        # Configurer l'affichage en fonction de la confiance
        if confidence < 0.8:
            result_label.config(text=f"⚠️ {class_name}\nConfidence: {confidence:.2%}", fg="orange")
        else:
            result_label.config(text=f"🛑 Prediction: {class_name}\nConfidence: {confidence:.2%}", fg="green")

        # Mettre à jour le compteur et le nom de fichier
        counter_label.config(text=f"Image {current_index + 1}/{len(image_files)}")
        filename = os.path.basename(file_path)
        filename_label.config(text=filename[:40] + "..." if len(filename) > 40 else filename) 

        # Gestion des boutons de navigation
        prev_btn["state"] = "normal" if current_index > 0 else "disabled"
        next_btn["state"] = "normal" if current_index < len(image_files) - 1 else "disabled"

# === Fonctions d'interface ===
def open_images():
    global image_files, current_index
    file_paths = filedialog.askopenfilenames(
        title="Sélectionnez des images de panneaux routiers",
        filetypes=[("Image files", ".jpg *.jpeg *.png *.bmp"), ("All files", ".*")]
    )
    if file_paths:
        image_files = list(file_paths)
        current_index = 0
        show_current_image()
        nav_frame.pack(pady=10)

def previous_image():
    global current_index
    if current_index > 0:
        current_index -= 1
        show_current_image()

def next_image():
    global current_index
    if current_index < len(image_files) - 1:
        current_index += 1
        show_current_image()

# === Interface Tkinter ===
root = tk.Tk()
root.title("🚦 Traffic Sign Classifier")
root.geometry("500x650")
root.config(bg="#f2f2f2")

# Widgets
title_label = tk.Label(root, text="Traffic Sign Recognition", font=("Helvetica", 18, "bold"), bg="#f2f2f2", fg="#333")
title_label.pack(pady=20)

upload_btn = tk.Button(root, text="📁 Upload Multiple Images", command=open_images,
                      font=("Helvetica", 14), bg="#007acc", fg="white",
                      padx=20, pady=10, bd=0, relief="raised", cursor="hand2")
upload_btn.pack(pady=10)

counter_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#f2f2f2", fg="#555")
counter_label.pack(pady=5)

filename_label = tk.Label(root, text="", font=("Helvetica", 10), bg="#f2f2f2", fg="#555")
filename_label.pack(pady=5)

image_label = tk.Label(root, bg="#f2f2f2")
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#f2f2f2", justify=tk.LEFT)
result_label.pack(pady=10)

nav_frame = tk.Frame(root, bg="#f2f2f2")
prev_btn = tk.Button(nav_frame, text="◀ Précédent", command=previous_image,
                     font=("Helvetica", 12), bg="#ddd", fg="#333",
                     padx=10, pady=5, state="disabled")
prev_btn.pack(side=tk.LEFT, padx=10)

next_btn = tk.Button(nav_frame, text="Suivant ▶", command=next_image,
                     font=("Helvetica", 12), bg="#ddd", fg="#333",
                     padx=10, pady=5, state="disabled")
next_btn.pack(side=tk.RIGHT, padx=10)

root.mainloop()