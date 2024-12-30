import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkFont
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk, ImageFont
import threading
import numpy as np
import random
import pickle
import os
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SimpleMLP:
    """Architektura sieci neuronowej"""
    def __init__(self, input_size, hidden_size, output_size, app_instance):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.app_instance = app_instance # instancja aplikacji (do zapisu danych z wykresu i progres bara), bo wiemy ze AlphabetRecognizerApp wywoluje SimpleMLP

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
        """Liczy wartosc funkcji sigmoidalnej dla zadanego argumentu"""
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        """Liczy pochodna z funkcji sigmoidalnej unipolarnej dla zadanego argumentu"""
        return x * (1 - x)

    def forward(self, inputs):
        """Wykonuje algorytm trenowania sieci """
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
        """Wkonuje algorytm wstecznej propagacji bledu w procesie trenowania sieci neuronowej"""
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

    def save_model(self, file_path):
        """Zapisuje model do pliku"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'weights_input_hidden': self.weights_input_hidden,
                'weights_hidden_output': self.weights_hidden_output,
                'bias_hidden': self.bias_hidden,
                'bias_output': self.bias_output,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'error_history': self.app_instance.error_history,
                'progress_var': self.app_instance.progress_var.get()
            }, f)
            
            print(self.app_instance.progress_var.get())
        print(f"Model zapisany do pliku: {file_path}")
    
    def load_model(self, file_path):
        """Wczytuje model z pliku"""
        self.app_instance.stop_training = True
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.weights_input_hidden = data['weights_input_hidden']
            self.weights_hidden_output = data['weights_hidden_output']
            self.bias_hidden = data['bias_hidden']
            self.bias_output = data['bias_output']
            self.input_size = data['input_size']
            self.hidden_size = data['hidden_size']
            self.output_size = data['output_size']
            self.app_instance.error_history = data.get('error_history', [])  # Odczyt historii błędów
            self.app_instance.progress_var.set(data.get('progress_var',0.0))  # Ustawienie paska
            self.app_instance.root.update_idletasks()  # Aktualizacja widoku paska progresu

        # Aktualizacja wykresu
        if self.app_instance.error_history:
            self.app_instance.update_plot(self.app_instance.error_history)
        print(f"Model wczytany z pliku: {file_path}")

class AlphabetRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Alphabet and Digit Recognizer")

        # Parametry segmentacji
        self.segment_size = 16
        self.font_path = os.path.join(os.path.dirname(__file__), "courier.ttf")  # Zamień na ścieżkę do Courier, jeśli dostępna
        self.font_size = 12
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        self.recognized_label = ""
        
        # Wymiary sieci i przygotowanie modelu
        input_size = 16 * 16
        hidden_size = 64
        output_size = 62  # 26 dużych liter, 26 małych liter i 10 cyfr
        self.model = SimpleMLP(input_size, hidden_size, output_size, app_instance=self)
        
        # Status treningu
        self.training_thread = None
        self.stop_training = False
        self.error_history = []
        
        # Mapa indeksów do etykiet wyjść
        self.labels = self.create_label_map()

        # Ustawienia grid layout
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=0, minsize=120)
        self.root.grid_rowconfigure(2, weight=0, minsize=60)
        self.root.grid_rowconfigure(3, weight=0)
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_rowconfigure(5, weight=0)  # Nowy wiersz na pasek postępu
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)
        self.root.grid_columnconfigure(4, weight=1)

        font = tkFont.Font(family="Helvetica", size=12, weight="bold")
            
        self.train_button = tk.Button(root, text="TRENUJ MODEL", font = font, command=self.start_training, borderwidth=3)
        
        self.train_button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.test_button = tk.Button(root, text="TESTUJ MODEL (Jedna Litera)", font = font, command=self.test_model, borderwidth=3)
        self.test_button.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.word_button = tk.Button(root, text="TESTUJ MODEL (Słowo)", font = font, command=self.on_generate, borderwidth=3)
        self.word_button.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # Przyciski do zapisywania i wczytywania modelu
        self.save_button = tk.Button(root, text="ZAPISZ STAN", font = font, command=self.save_model, borderwidth=3)
        self.save_button.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")

        self.load_button = tk.Button(root, text="WCZYTAJ STAN", font = font, command=self.load_model, borderwidth=3)
        self.load_button.grid(row=0, column=4, padx=5, pady=5, sticky="nsew")
     
        # Przyciski i etykiety
        self.label_frame = ttk.Frame(root)
        self.label_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Tworzenie widgetu tk.Text
        self.label_text = tk.Text(self.label_frame, font=font, relief="groove", wrap="word", height=5)
        self.label_text.pack(side="left", fill="both", expand=True)

        # Dodanie suwaka przewijania
        self.scrollbar = ttk.Scrollbar(self.label_frame, orient="vertical", command=self.label_text.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.update_label("ROZPOZNANO:")

        # Powiązanie suwaka z widgetem tk.Text
        self.label_text.config(yscrollcommand=self.scrollbar.set)

        # Ustawienie trybu tylko do odczytu (opcjonalne)
        self.label_text.bind("<Key>", lambda e: "break")  # Zablokuj edycję

        ## Przyciski i etykiety
        #self.label = tk.Label(root, text="ROZPOZNANO: ", font = font, relief="groove")
        #self.label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Text widget z pionowym suwakiem
        self.text_frame = ttk.Frame(self.root)
        self.text_frame.grid(row=1, column=2, columnspan=4, padx=5, pady=5, sticky="nsew")

        self.text_input = tk.Text(self.text_frame, wrap="word", height=5, width=50, font=("Courier", 12))
        self.text_input.grid(row=0, column=0, sticky="nsew")

        # Suwak pionowy
        self.scrollbar = ttk.Scrollbar(self.text_frame, orient="vertical", command=self.text_input.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.text_input.configure(yscrollcommand=self.scrollbar.set)

        # Konfiguracja kolumn i wierszy w ramce
        self.text_frame.columnconfigure(0, weight=1)
        self.text_frame.rowconfigure(0, weight=1)

        ## Etykieta do wyświetlania obrazu
        #self.text_input = ttk.Entry(self.root, width=50, font=("Courier", 12))
        #self.text_input.grid(row=1, column=2, columnspan=4, padx=5, pady=5, sticky="nsew")

        # Etykieta do wyświetlania obrazu
        self.image_label = tk.Label(root, bg="white", bd=4, relief="groove")
        self.image_label.grid(row=2, column=0, columnspan=5, padx=5, pady=5, sticky="nsew")
        
        # Ramka na wyświetlenie segmentowanego obrazu
        self.segment_display_frame = tk.Frame(root, borderwidth=3)
        self.segment_display_frame.grid(row=2, column=0, columnspan=5, padx=5, pady=5, sticky="nsew")

        self.load_label = tk.Label(root, text="")
        self.load_label.grid(row=3, column=0, columnspan=5, padx=5, pady=5, sticky="w")

        # Wykres błędu
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Średni błąd w funkcji iteracji")
        self.ax.set_xlabel("Iteracje")
        self.ax.set_ylabel("Średni błąd")
        self.ax.grid(True)  # Włączenie siatki
        self.error_line, = self.ax.plot([], [], 'r-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(highlightthickness=2, highlightbackground="black")
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=5, padx=5, pady=5, sticky="nsew")

        # Przyciski i etykiety
        self.label2 = tk.Label(root, text="POZIOM TRENINGU:", font = font, relief="groove")
        self.label2.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Dodanie Progressbar do GUI
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=2, columnspan=3, padx=5, pady=5, sticky="nsew")


        # Obsługa zamknięcia aplikacji
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
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
            self.load_label.config(text=f"ZAŁADOWANY STAN: {os.path.basename(file_path)}")
            messagebox.showinfo("Sukces", f"Model wczytano z pliku: {file_path}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać modelu: {e}")

    def create_label_map(self):
        """Tworzy nazwy dla danych testowych"""
        labels = []
        # Duże litery
        labels.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        # Małe litery
        labels.extend([f".{chr(i)}" for i in range(ord('a'), ord('z') + 1)])
        # Cyfry
        labels.extend([str(i) for i in range(10)])
        return labels

    def load_image(self, file_path):
        """laduje obrazek"""
        img = Image.open(file_path).convert('L').resize((16, 16))
        img = img.point(lambda p: 1 if p < 128 else 0)
        return np.array(img).flatten().tolist()
    
    def test_word(self):
        """Wczytuje obrazek i rozpoznaje slowo w nim umieszczone"""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            recognized_word = self.segment_and_recognize_word(file_path)
            self.label.config(text=f"Rozpoznane zdanie: {recognized_word}")

    def start_training(self):
        """Trenuje siec neuronowa, dzieli te aktywnosc na osobny watek, aktualizuje wykres i progres bar"""
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Trening w toku", "Sieć jest w trakcie treningu.")
            return

        # Ustawienie flagi dla nowego treningu
        self.stop_training = False
        self.error_history.clear()
        training_data = self.generate_training_data()

        # Funkcja do treningu w wątku
        def training_thread():
            self.train_button.config(state=tk.DISABLED)
            self.progress_var.set(0)  # Reset paska postępu
            epochs = 600  # Liczba epok
            learning_rate = 0.1

            # Trening w pętli
            for epoch in range(epochs):
                if self.stop_training:
                    print("Trening został przerwany.")
                    break

                error_sum = 0
                for inputs, expected_output in training_data:
                    outputs = self.model.forward(inputs)
                    self.model.backward(inputs, expected_output, learning_rate)
                    error_sum += sum((expected_output[k] - outputs[k]) ** 2 for k in range(len(expected_output)))

                mean_error = error_sum / len(training_data)
                if epoch == 0:
                    print(f"Epoch: {epoch+1}, Mean Error: {mean_error:.4f}") # tylko 1 raz
                    self.error_history.append((epoch+1, mean_error))
                    self.update_plot(self.error_history)
                # Aktualizacja wykresu co 50 iteracji
                elif epoch % 50 == 49:   
                    print(f"Epoch: {epoch+1}, Mean Error: {mean_error:.4f}")
                    self.error_history.append((epoch+1, mean_error))
                    self.update_plot(self.error_history)

                # Aktualizacja progres bara co 10 iteracji
                if epoch % 10 == 9:
                    progress = (epoch + 1) / epochs * 100
                    self.progress_var.set(progress)
                    self.root.update_idletasks()

            self.train_button.config(state=tk.NORMAL)
            self.training_thread = None
            messagebox.showinfo("Trening zakończony", "Model został wytrenowany.")

        # Uruchomienie wątku treningowego w trybie daemon
        self.training_thread = threading.Thread(target=training_thread, daemon=True)
        self.training_thread.start()

    def update_plot(self, error_history):
        """"Aktualizuje wykres"""
        # Aktualizacja danych na wykresie
        epochs, errors = zip(*error_history)
        self.error_line.set_data(epochs, errors)
        self.ax.set_xlim(0, max(epochs) + 50)
        self.ax.set_ylim(0, max(errors) + 0.1)
        self.canvas.draw()

    def generate_training_data(self):
        """Wskazuje zbior danych treningowych"""
        training_data = []
        print(os.path.dirname(__file__))
        base_path = os.path.join(os.path.dirname(__file__), 'dane15')  # Zakładamy, że folder "Data" jest w tym samym folderze co plik .py
        

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
                    output_data = [1 if i == label_idx else 0 for i in range(len(self.labels))]
                    training_data.append((input_data, output_data))
        return training_data

    def test_model(self):
        """Wczytuje obraz i rozpoznaje na nim znak"""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

             # Wczytanie obrazu
        img = Image.open(file_path).resize((100, 100))  # Zmieniamy rozmiar do 100x100
        img_tk = ImageTk.PhotoImage(img)
        
        # Wyświetlenie obrazu w aplikacji
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Przechowujemy referencję, aby nie został zgarnięty przez garbage collector
        
        input_data = self.load_image(file_path)
        output = self.model.forward(input_data)
        recognized_index = output.index(max(output))
        recognized_label = self.labels[recognized_index].replace(".", "")
        self.update_label(f"ROZPOZNANO: {recognized_label}")

    def on_generate(self):
        """Obsługuje kliknięcie przycisku generowania obrazu."""
        input_text = self.text_input.get("1.0", "end").strip()
        if not input_text:
            messagebox.showwarning("Uwaga", "Pole tekstowe nie może być puste!")
            return

        # Czyszczenie poprzednich obrazów w widoku
        #for widget in self.image_frame.winfo_children():
        #    widget.destroy()

        # Generowanie obrazów dla każdego znaku
        for idx, letter in enumerate(input_text):
            if letter.isspace():  # Pomijamy znaki spacji
                if letter == "\n":
                    self.recognized_label += "\n"
                else:
                    self.recognized_label += " "
                continue
            
            img = self.create_segment_image(letter)
            img_tk = ImageTk.PhotoImage(img)
            #label = tk.Label(self.image_frame, image=img_tk)
            #label.image = img_tk  # Referencja, aby obraz nie został usunięty przez garbage collector
            #label.grid(row=idx // 50, column=idx % 50, padx=1, pady=1)  # 50 znaków na linię

            # Generowanie binarnej reprezentacji
            binary_matrix = self.get_binary_representation(img)
            pixels = []

            for row in binary_matrix:
                for pixel in row:
                    pixels.append(0 if pixel == 1 else 255)  # 0 = czarny, 255 = biały

            img.putdata(pixels)
            # Przygotowanie wejścia do sieci
            input_data = np.array(img.point(lambda p: 1 if p < 128 else 0)).flatten().tolist()
            output = self.model.forward(input_data)  # Przekazanie danych do sieci
            recognized_index = output.index(max(output))
            self.recognized_label += self.labels[recognized_index].replace(".", "")
        self.update_label(f"ROZPOZNANO:\n {self.recognized_label}")
        self.recognized_label = ""

    def create_segment_image(self, letter):
        """
        Tworzy obraz 16x16 z pojedynczym znakiem.
        """
        img = Image.new("1", (self.segment_size, self.segment_size), 1)  # "1" oznacza czarno-biały obraz
        draw = ImageDraw.Draw(img)
        w, h = self.font.getbbox(letter)[2], self.font.getbbox(letter)[3]
        x = (self.segment_size - w) // 2
        y = (self.segment_size - h) // 2
        draw.text((x, y), letter, fill=0, font=self.font)  # 0 to czarny kolor
        img.show

        return img

    def get_binary_representation(self, img):
        """
        Konwertuje obraz na binarną reprezentację (0 - biały, 1 - czarny).
        """
        img = img.convert("1")  # Upewniamy się, że obraz jest w trybie czarno-białym
        pixels = list(img.getdata())  # Pobieramy dane pikseli
        binary_matrix = []

        for i in range(self.segment_size):
            row = pixels[i * self.segment_size:(i + 1) * self.segment_size]
            binary_matrix.append([1 if pixel == 0 else 0 for pixel in row])  # 0 to czarny, 1 to biały

        return binary_matrix

    def update_label(self, text):
        self.label_text.config(state="normal")  # Odblokowanie edycji
        self.label_text.delete("1.0", "end")   # Wyczyść istniejący tekst
        self.label_text.insert("1.0", text)    # Wstaw nowy tekst
        self.label_text.config(state="disabled")  # Zablokuj ponownie

    def test_word(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            recognized_word = self.segment_and_recognize_word(file_path)
            self.label.config(text=f"ROZPOZNANO: {recognized_word}")

    def on_closing(self):
        """Wywolywana gdy aplikacja sie zamyka, niszczy dzialajacy w tle watek treningowy sieci neuronowej"""
        # Ustawienie flagi stop_training na True, aby przerwać wątek treningowy
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training = True  # Ustawienie flagi przerwania
            self.root.after(0, self.check_training_thread)  
        else:
            self.root.destroy()

    def check_training_thread(self):
        """Sprawdza czy watek treningowy jest aktywny"""
        if self.training_thread and self.training_thread.is_alive():
            self.root.after(50, self.check_training_thread) # Sprawdzanie co 50 ms, czy wątek się zakończył
        else:
            self.root.destroy()

# Uruchomienie aplikacji
root = tk.Tk()
window_width = 1320
window_height = 680
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
app = AlphabetRecognizerApp(root)
root.mainloop()