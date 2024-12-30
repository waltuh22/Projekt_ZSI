import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import random
import os
import threading
import shutil
from PIL import ImageDraw, ImageFont
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.ttk import Progressbar

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
        self.bias_hidden = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-0.5, 0.5) for _ in range(output_size)]

        self.hidden_layer = [0] * hidden_size
        self.output_layer = [0] * output_size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        for j in range(self.hidden_size):
            activation = sum(inputs[i] * self.weights_input_hidden[i][j] for i in range(self.input_size)) + self.bias_hidden[j]
            self.hidden_layer[j] = self.sigmoid(activation)

        for k in range(self.output_size):
            activation = sum(self.hidden_layer[j] * self.weights_hidden_output[j][k] for j in range(self.hidden_size)) + self.bias_output[k]
            self.output_layer[k] = self.sigmoid(activation)

        return self.output_layer

    def backward(self, inputs, expected_output, learning_rate):
        output_deltas = [(expected_output[k] - self.output_layer[k]) * self.sigmoid_derivative(self.output_layer[k])
                         for k in range(self.output_size)]

        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.weights_hidden_output[j][k] += learning_rate * output_deltas[k] * self.hidden_layer[j]
        for k in range(self.output_size):
            self.bias_output[k] += learning_rate * output_deltas[k]

        hidden_deltas = [sum(output_deltas[k] * self.weights_hidden_output[j][k] for k in range(self.output_size))
                         * self.sigmoid_derivative(self.hidden_layer[j]) for j in range(self.hidden_size)]

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += learning_rate * hidden_deltas[j] * inputs[i]
        for j in range(self.hidden_size):
            self.bias_hidden[j] += learning_rate * hidden_deltas[j]

    def train(self, training_data, epochs=20, learning_rate=0.1, stop_flag=None, error_callback=None, progress_callback=None):
        error_history = []
        for epoch in range(epochs):
            if stop_flag and stop_flag():
                print("Trening został przerwany.")
                break
            error_sum = 0
            for inputs, expected_output in training_data:
                outputs = self.forward(inputs)
                self.backward(inputs, expected_output, learning_rate)
                error_sum += sum((expected_output[k] - outputs[k]) ** 2 for k in range(len(expected_output)))

            mean_error = error_sum / len(training_data)
            if epoch % 2 == 0:
                error_history.append((epoch, mean_error))
                if error_callback:
                    error_callback(error_history)
            if progress_callback:
                progress_callback(epoch + 1, epochs)

class AlphabetRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Alphabet Recognizer")
        self.input_size = 24 * 24
        self.hidden_size = 128
        self.output_size = 62
        self.model = SimpleMLP(self.input_size, self.hidden_size, self.output_size)
        self.labels = self.create_label_map()
        self.training_in_progress = False

        self.input_field = tk.Entry(root, font=("Arial", 14), justify="center")
        self.input_field.pack(pady=10, fill=tk.X, padx=20)

        self.recognize_button = tk.Button(root, text="Rozpoznaj", command=self.recognize_input)
        self.recognize_button.pack(pady=10)

        self.output_label = tk.Label(root, text="Rozpoznane: ", font=("Arial", 16))
        self.output_label.pack(pady=10)

        # Miejsce na obrazy
        self.image_canvas = tk.Canvas(root, height=50, bg="white")
        self.image_canvas.pack(pady=10, fill=tk.BOTH, expand=True)

        self.train_button = tk.Button(root, text="Trenuj", command=self.start_training)
        self.train_button.pack(pady=10)

        self.progress = Progressbar(root, orient="horizontal", mode="determinate", length=300)
        self.progress.pack(pady=10)

        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Średni błąd treningu")
        self.ax.set_xlabel("Epoki")
        self.ax.set_ylabel("Średni błąd")
        self.error_plot = FigureCanvasTkAgg(self.figure, root)
        self.error_plot.get_tk_widget().pack(pady=10)

        self.stop_training_flag = False

    def create_label_map(self):
        labels = []
        labels.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        labels.extend([chr(i).upper() for i in range(ord('a'), ord('z') + 1)])
        labels.extend([str(i) for i in range(10)])
        return labels

    def load_image(self, image):
        img = image.convert('L').resize((24, 24))
        img = img.point(lambda p: 1 if p < 128 else 0)
        return np.array(img).flatten().tolist()

    def recognize_input(self):
        # Rozpoznanie tekstu i wyświetlenie obrazów
        input_text = self.input_field.get()
        if not input_text:
            messagebox.showwarning("Błąd", "Pole tekstowe jest puste.")
            return

        recognized_text = ""
        self.image_canvas.delete("all")  # Wyczyść poprzednie obrazy

        for idx, char in enumerate(input_text):
            char_image = self.generate_character_image(char)

            # Wyświetl obraz znaku w Canvas
            char_image_resized = char_image.resize((24, 24))
            char_photo = ImageTk.PhotoImage(char_image)
            self.image_canvas.create_image(idx * 70 + 10, 10, anchor="nw", image=char_photo)
            self.image_canvas.image = char_photo  # Zachowanie referencji

            # Rozpoznanie znaku
            input_data = self.load_image(char_image)
            output = self.model.forward(input_data)
            recognized_index = output.index(max(output))
            recognized_char = self.labels[recognized_index]
            recognized_text += recognized_char

        self.output_label.config(text=f"Rozpoznane: {recognized_text}")

    def generate_character_image(self, char):
        img_size = (24, 24)
        font_size = 14
        image = Image.new("RGB", img_size, color="white")
        draw = ImageDraw.Draw(image)

        try:
            font_path = "arial.ttf"
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        bbox = font.getbbox(char)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_position = ((img_size[0] - text_width) // 2, (img_size[1] - text_height) // 2)
        draw.text(text_position, char, font=font, fill="black")
        return image

    def start_training(self):
        if self.training_in_progress:
            return

        self.train_button.config(state=tk.DISABLED)
        self.training_in_progress = True
        training_data = self.generate_training_data()

        threading.Thread(
            target=lambda: self.model.train(
                training_data,
                epochs=20,
                learning_rate=0.1,
                stop_flag=lambda: self.stop_training_flag,
                error_callback=self.update_error_plot,
                progress_callback=self.update_progress
            ),
            daemon=True
        ).start()

    def update_error_plot(self, error_history):
        epochs, errors = zip(*error_history)
        self.ax.clear()
        self.ax.plot(epochs, errors, label="Średni błąd", color="blue")
        self.ax.set_title("Średni błąd treningu")
        self.ax.set_xlabel("Epoki")
        self.ax.set_ylabel("Średni błąd")
        self.ax.legend()
        self.error_plot.draw()

    def update_progress(self, current_epoch, total_epochs):
        self.progress["maximum"] = total_epochs
        self.progress["value"] = current_epoch
        if current_epoch == total_epochs:
            self.train_button.config(state=tk.NORMAL)
            self.training_in_progress = False

    def generate_training_data(self):
        training_data = []
        base_path = "dane1"
        for label_idx, label in enumerate(self.labels):
            folder_path = os.path.join(base_path, f".{label}")
            if not os.path.exists(folder_path):
                continue

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                image = Image.open(file_path)
                inputs = self.load_image(image)
                expected_output = [0] * self.output_size
                expected_output[label_idx] = 1
                training_data.append((inputs, expected_output))

        return training_data


root = tk.Tk()
app = AlphabetRecognizerApp(root)
root.mainloop()
