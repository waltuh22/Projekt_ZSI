import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
import threading
import numpy as np
import random
import math
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicjalizacja wag losowymi wartościami
        self.weights_input_hidden = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]

        # Inicjalizacja biasów losowymi wartościami
        self.bias_hidden = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-0.5, 0.5) for _ in range(output_size)]

        # Przechowywanie ostatnich wartości dla propagacji wstecznej
        self.hidden_layer = [0] * hidden_size
        self.output_layer = [0] * output_size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        # Warstwa ukryta
        for j in range(self.hidden_size):
            activation = sum(inputs[i] * self.weights_input_hidden[i][j] for i in range(self.input_size)) + self.bias_hidden[j]
            self.hidden_layer[j] = self.sigmoid(activation)

        # Warstwa wyjściowa
        for k in range(self.output_size):
            activation = sum(self.hidden_layer[j] * self.weights_hidden_output[j][k] for j in range(self.hidden_size)) + self.bias_output[k]
            self.output_layer[k] = self.sigmoid(activation)

        return self.output_layer

    def backward(self, inputs, expected_output, learning_rate):
        output_deltas = [(expected_output[k] - self.output_layer[k]) * self.sigmoid_derivative(self.output_layer[k])
                         for k in range(self.output_size)]

        # Aktualizacja wag ukryta-wyjście
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.weights_hidden_output[j][k] += learning_rate * output_deltas[k] * self.hidden_layer[j]
        for k in range(self.output_size):
            self.bias_output[k] += learning_rate * output_deltas[k]

        # Błędy dla warstwy ukrytej
        hidden_deltas = [sum(output_deltas[k] * self.weights_hidden_output[j][k] for k in range(self.output_size))
                         * self.sigmoid_derivative(self.hidden_layer[j]) for j in range(self.hidden_size)]

        # Aktualizacja wag wejście-ukryta
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += learning_rate * hidden_deltas[j] * inputs[i]
        for j in range(self.hidden_size):
            self.bias_hidden[j] += learning_rate * hidden_deltas[j]

    
    def train(self, training_data, epochs=200, learning_rate=0.1, stop_flag=None, error_callback=None):
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
            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, Mean Error: {mean_error:.4f}")
                error_history.append((epoch, mean_error))
                if error_callback:
                    error_callback(error_history)  # Zaktualizowanie wykresu
class AlphabetRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Alphabet and Digit Recognizer")

        # Wymiary sieci i przygotowanie modelu
        input_size = 16 * 16
        hidden_size = 64
        output_size = 62  # 26 dużych liter, 26 małych liter i 10 cyfr
        self.model = SimpleMLP(input_size, hidden_size, output_size)

        # Status treningu
        self.training_thread = None
        self.stop_training = False
        self.error_history = []

        # Mapa indeksów do etykiet wyjść
        self.labels = self.create_label_map()

        # Przygotowanie interfejsu użytkownika
        self.setup_ui()

        # Obsługa zamknięcia aplikacji
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_label_map(self):
        labels = []
        labels.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        labels.extend([f".{chr(i)}" for i in range(ord('a'), ord('z') + 1)])
        labels.extend([str(i) for i in range(10)])
        return labels

    def setup_ui(self):
        # Główna ramka
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Pole tekstowe do wpisania słowa
        self.word_entry_label = ttk.Label(self.main_frame, text="Wpisz słowo:")
        self.word_entry_label.grid(row=0, column=0, sticky=tk.W)

        self.word_entry = ttk.Entry(self.main_frame)
        self.word_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.word_entry.bind("<Return>", lambda event: self.recognize_word())

        # Przycisk rozpoznawania słowa
        self.recognize_button = ttk.Button(self.main_frame, text="Rozpoznaj Słowo", command=self.recognize_word)
        self.recognize_button.grid(row=0, column=2, sticky=tk.E, padx=5)

        # Wyświetlanie wyniku
        self.result_label = ttk.Label(self.main_frame, text="Rozpoznane słowo:")
        self.result_label.grid(row=1, column=0, columnspan=3, sticky=tk.W)

        self.result_display = ttk.Label(self.main_frame, text="", background="white", relief="sunken", anchor="w")
        self.result_display.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=5)

        # Segmenty wyświetlane w ramce
        self.segment_display_frame = ttk.Frame(self.main_frame)
        self.segment_display_frame.grid(row=3, column=0, columnspan=3, sticky=tk.NSEW)

        # Responsywność układu
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(3, weight=1)

        # Przycisk do treningu modelu
        self.train_button = ttk.Button(self.main_frame, text="Trenuj Model", command=self.start_training)
        self.train_button.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=5)

        # Wykres błędu
        self.fig = Figure(figsize=(5, 2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Średni błąd w funkcji iteracji")
        self.ax.set_xlabel("Iteracje")
        self.ax.set_ylabel("Średni błąd")
        self.error_line, = self.ax.plot([], [], 'r-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=3, sticky=tk.EW)

    def recognize_word(self):
        word = self.word_entry.get().strip()
        if not word:
            messagebox.showwarning("Błąd", "Pole tekstowe nie może być puste!")
            return

        # Usunięcie wcześniejszych obrazów
        for widget in self.segment_display_frame.winfo_children():
            widget.destroy()

        # Generowanie i wyświetlanie liter
        for char in word:
            image = self.generate_image(char)
            img_display = ImageTk.PhotoImage(image)
            label = tk.Label(self.segment_display_frame, image=img_display)
            label.image = img_display  # Przechowywanie referencji
            label.pack(side=tk.LEFT, padx=5, pady=5)

    def generate_image(self, char):
        """Generuje obraz z daną literą."""
        img_size = (24, 24)  # Rozmiar obrazu
        font_size = 14       # Rozmiar czcionki

        # Tworzenie nowego obrazu
        image = Image.new("RGB", img_size, color="white")
        draw = ImageDraw.Draw(image)

        # Ustawienie czcionki
        try:
            font_path = "arial.ttf"  # Ścieżka do czcionki
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # Wyśrodkowanie tekstu
        bbox = font.getbbox(char)  # Pobiera prostokąt otaczający znak
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_position = ((img_size[0] - text_width) // 2, (img_size[1] - text_height) // 2)

        draw.text(text_position, char, font=font, fill="black")
        return image


    def recognize_character(self, char_image):
        resized = char_image.resize((16, 16)).point(lambda p: 1 if p < 128 else 0)
        input_data = np.array(resized).flatten().tolist()
        output = self.model.forward(input_data)
        recognized_index = output.index(max(output))
        return self.labels[recognized_index].replace(".", "")
    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Trening w toku", "Sieć jest w trakcie treningu.")
            return

        # Ustawienie flagi dla nowego treningu
        self.stop_training = False
        self.error_history.clear()
        training_data = self.generate_image()

        # Funkcja do treningu w wątku
        def training_thread():
            self.train_button.config(state=tk.DISABLED)
            self.model.train(training_data, stop_flag=lambda: self.stop_training, error_callback=self.update_plot)
            self.train_button.config(state=tk.NORMAL)
            self.training_thread = None
            messagebox.showinfo("Trening zakończony", "Model został wytrenowany.")

        # Uruchomienie wątku treningowego w trybie daemon
        self.training_thread = threading.Thread(target=training_thread, daemon=True)
        self.training_thread.start()
    def on_closing(self):
        # Ustawienie flagi stop_training na True, aby przerwać wątek treningowy
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training = True  # Ustawienie flagi przerwania
            self.root.after(0, self.check_training_thread)  
        else:
            self.root.destroy()

    def check_training_thread(self):
        if self.training_thread and self.training_thread.is_alive():
            self.root.after(50, self.check_training_thread) # Sprawdzanie co 50 ms, czy wątek się zakończył
        else:
            self.root.destroy()
# Uruchomienie aplikacji
root = tk.Tk()
window_width, window_height = 600, 400
screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
x, y = (screen_width - window_width) // 2, (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
app = AlphabetRecognizerApp(root)
root.mainloop()