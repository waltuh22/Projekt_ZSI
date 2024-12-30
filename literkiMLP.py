import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import numpy as np
import random
import math
import pickle
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
        return 1 / (1 + math.exp(-x))

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
        # Obliczenie błędów dla warstwy wyjściowej
        output_deltas = [(expected_output[k] - self.output_layer[k]) * self.sigmoid_derivative(self.output_layer[k])
                         for k in range(self.output_size)]

        # Aktualizacja wag i biasów ukryta-wyjście
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.weights_hidden_output[j][k] += learning_rate * output_deltas[k] * self.hidden_layer[j]
        for k in range(self.output_size):
            self.bias_output[k] += learning_rate * output_deltas[k]

        # Błędy dla warstwy ukrytej
        hidden_deltas = [sum(output_deltas[k] * self.weights_hidden_output[j][k] for k in range(self.output_size))
                         * self.sigmoid_derivative(self.hidden_layer[j]) for j in range(self.hidden_size)]

        # Aktualizacja wag i biasów wejście-ukryta
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += learning_rate * hidden_deltas[j] * inputs[i]
        for j in range(self.hidden_size):
            self.bias_hidden[j] += learning_rate * hidden_deltas[j]


    def train(self, training_data, epochs=200, learning_rate=0.1, stop_flag=None, update_error_callback=None):
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
                if update_error_callback:
                    update_error_callback(error_history)  # Zaktualizowanie wykresu
    def save_model(self, file_path):
        """Zapisuje model do pliku."""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'weights_input_hidden': self.weights_input_hidden,
                'weights_hidden_output': self.weights_hidden_output,
                'bias_hidden': self.bias_hidden,
                'bias_output': self.bias_output,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
            }, f)
        print(f"Model zapisany do pliku: {file_path}")
    
    def load_model(self, file_path):
        """Wczytuje model z pliku."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.weights_input_hidden = data['weights_input_hidden']
            self.weights_hidden_output = data['weights_hidden_output']
            self.bias_hidden = data['bias_hidden']
            self.bias_output = data['bias_output']
            self.input_size = data['input_size']
            self.hidden_size = data['hidden_size']
            self.output_size = data['output_size']
        print(f"Model wczytany z pliku: {file_path}")

class AlphabetRecognizerApp:
    def __init__(self, ROOT):
        self.ROOT = ROOT
        self.ROOT.title("Alphabet Recognizer")
        
        # Wymiary sieci i przygotowanie modelu
        input_size = 16 * 16
        hidden_size = 64
        output_size = 62  # 26 liter alfabetu
        self.model = SimpleMLP(input_size, hidden_size, output_size)
        
        # Status treningu
        self.training_thread = None
        self.stop_training = False
        
        # Mapa indeksów do etykiet wyjść
        self.labels = self.create_label_map()

        ## Ustawienia grid layout
        self.ROOT.grid_rowconfigure(0, weight=0)
        self.ROOT.grid_rowconfigure(1, weight=0)
        self.ROOT.grid_rowconfigure(2, weight=0, minsize=120)
        self.ROOT.grid_rowconfigure(3, weight=0)
        self.ROOT.grid_columnconfigure(0, weight=1)
        self.ROOT.grid_columnconfigure(1, weight=1)

        # Przyciski i etykiety
        self.label = tk.Label(ROOT, text="Rozpoznaję: ", relief="groove")
        self.label.grid(row=1, column=0, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.train_button = tk.Button(ROOT, text="Trenuj Model", command=self.start_training, borderwidth=3)
        self.train_button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.test_button = tk.Button(ROOT, text="Testuj Model", command=self.test_model, borderwidth=3)
        self.test_button.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Przyciski do zapisywania i wczytywania modelu
        self.save_button = tk.Button(ROOT, text="Zapisz Model", command=self.save_model, borderwidth=3)
        self.save_button.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        self.load_button = tk.Button(ROOT, text="Wczytaj Model", command=self.load_model, borderwidth=3)
        self.load_button.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

        # Etykieta do wyświetlania obrazu
        self.image_label = tk.Label(ROOT, bg="white", bd=4, relief="groove")
        self.image_label.grid(row=1, column=1,rowspan=2, columnspan=3, padx=5, pady=5, sticky="nsew")

        # Wykres błędu
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Średni błąd w funkcji iteracji")
        self.ax.set_xlabel("Iteracje")
        self.ax.set_ylabel("Średni błąd")
        self.error_line, = self.ax.plot([], [], 'r-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=ROOT)

        # Pogrubienie ramki canvasu
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(highlightthickness=2, highlightbackground="black")
        canvas_widget.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

        # Obsługa zamknięcia aplikacji
        self.ROOT.protocol("WM_DELETE_WINDOW", self.on_closing)

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pliki modelu", "*.pkl")])
        if not file_path:
            return
        try:
            self.model.save_model(file_path)
            messagebox.showinfo("Sukces", f"Model zapisano do pliku: {file_path}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się zapisać modelu: {e}")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pliki modelu", "*.pkl")])
        if not file_path:
            return
        try:
            self.model.load_model(file_path)
            messagebox.showinfo("Sukces", f"Model wczytano z pliku: {file_path}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać modelu: {e}")

    def create_label_map(self):
        labels = []
        # Duże litery
        labels.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        # Małe litery
        labels.extend([f".{chr(i)}" for i in range(ord('a'), ord('z') + 1)])
        # Cyfry
        labels.extend([str(i) for i in range(10)])
        return labels

    def load_image(self, file_path):
        img = Image.open(file_path).convert('L').resize((16, 16))
        # Obraz trybu "L", oznacza, że jest to obraz jednokanałowy - zwykle interpretowany jako skala szarości.
        # L oznacza to, że przechowuje tylko Luminancję. Jest bardzo kompaktowy, ale przechowuje tylko skalę szarości, a nie kolor.
        img = img.point(lambda p: 1 if p < 128 else 0) # binaryzacja

        return np.array(img).flatten().tolist()

    def generate_training_data(self):
        training_data = []
        print(os.path.dirname(__file__))
        base_path = os.path.join(os.path.dirname(__file__), 'Dane')  # Zakładamy, że folder "Data" jest w tym samym folderze co plik .py
        

        # Sprawdzenie, czy folder 'Data' istnieje
        if not os.path.exists(base_path):
            messagebox.showerror("Błąd", "Folder 'Data' nie istnieje.")
            return training_data

        training_data = []
        for label_idx, label in enumerate(self.labels):
            folder_path = os.path.join(base_path, label)
            if os.path.isdir(folder_path):
                for image_file in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_file)
                    input_data = self.load_image(image_path)
                    expected_output = [0] * 62
                    expected_output[label_idx] = 1
                    training_data.append((input_data, expected_output))
        return training_data

    def update_plot(self, error_history):
        # Aktualizacja danych na wykresie
        epochs, errors = zip(*error_history)
        self.error_line.set_data(epochs, errors)
        self.ax.set_xlim(0, max(epochs) + 50)
        self.ax.set_ylim(0, max(errors) + 0.1)
        self.canvas.draw()

    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Trening w toku", "Sieć jest w trakcie treningu.")
            return

        # Ustawienie flagi dla nowego treningu
        self.stop_training = False
        training_data = self.generate_training_data() # self wskazuje na te konkretna klase (troche jak .this w javie)
        # chcemy do zmiennej lokalnej jaka jest training_data zapisac wynik metody generate training data z tej klasy
        if (training_data == []):
            return
        # Funkcja do treningu w wątku
        def training_thread():
            self.train_button.config(state=tk.DISABLED)
            self.model.train(training_data, stop_flag=lambda: self.stop_training, update_error_callback=self.update_plot)
            self.training_thread = None
            self.train_button.config(state=tk.NORMAL)
            messagebox.showinfo("Trening zakończony", "Model został wytrenowany.")

        # Uruchomienie wątku treningowego
        self.training_thread = threading.Thread(target=training_thread, daemon=True)
        self.training_thread.start()

    def test_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
        if not file_path:
            return
        
         # Wczytanie obrazu
        img = Image.open(file_path).resize((100, 100))  # Zmieniamy rozmiar do 100x100
        img_tk = ImageTk.PhotoImage(img)
        
        # Wyświetlenie obrazu w aplikacji
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Przechowujemy referencję, aby nie został zgarnięty przez garbage collector

        # Wczytanie danych i rozpoznanie
        input_data = self.load_image(file_path)
        output = self.model.forward(input_data)
        recognized_index = output.index(max(output))
        recognized_label = self.labels[recognized_index]
        self.label.config(text=f"Rozpoznaję: {recognized_label}")

    def on_closing(self):
        # Ustawienie flagi stop_training na True, aby przerwać wątek treningowy
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training = True  # Ustawienie flagi przerwania
            self.ROOT.after(0, self.check_training_thread)  
        else:
            self.ROOT.destroy()

    def check_training_thread(self):
        if self.training_thread and self.training_thread.is_alive():
            self.ROOT.after(50, self.check_training_thread) # Sprawdzanie co 50 ms, czy wątek się zakończył
        else:
            self.ROOT.destroy()

ROOT = tk.Tk()
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 570
screen_width = ROOT.winfo_screenwidth()
screen_height = ROOT.winfo_screenheight()
x = (screen_width // 2) - (WINDOW_WIDTH // 2)
y = (screen_height // 2) - (WINDOW_HEIGHT // 2)
ROOT.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
app = AlphabetRecognizerApp(ROOT)
ROOT.mainloop()