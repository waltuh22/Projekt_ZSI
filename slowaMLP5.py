import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentacja Obrazu (16x16)")
        self.root.geometry("800x600")

        # Parametry segmentacji
        self.segment_size = 16
        self.font_path = os.path.join(os.path.dirname(__file__), "courier.ttf")  # Zamień na ścieżkę do Courier, jeśli dostępna
        self.font_size = 12
        self.font = ImageFont.truetype(self.font_path, self.font_size)

        # Interfejs użytkownika
        self._create_widgets()

    def _create_widgets(self):
        # Pole tekstowe do wprowadzania liter
        text_input_label = ttk.Label(self.root, text="Wpisz tekst do segemntacji:")
        text_input_label.pack(pady=10)

        self.text_input = ttk.Entry(self.root, width=50, font=("Courier", 12))
        self.text_input.pack(pady=5)

        # Przycisk generowania obrazu
        generate_button = ttk.Button(self.root, text="Generuj Obraz", command=self.on_generate)
        generate_button.pack(pady=10)

        # Kontener na segmenty obrazu
        #self.image_frame = tk.Frame(self.root)
        #self.image_frame.pack(pady=10)

        # Pole tekstowe do wyświetlania informacji o segmentach
        # segment_info_label = ttk.Label(self.root, text="Informacje o segmentach:")
        # segment_info_label.pack(pady=10)

        # self.segment_info_text = tk.Text(self.root, width=50, height=10, font=("Courier", 12))
        # self.segment_info_text.pack(pady=5)

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

    def save_binary_image(self, letter, binary_matrix):
        """
        Zapisuje binarną reprezentację obrazu do pliku .png.
        Tworzy folder <nazwa_znaku> w folderze 'dane15' i zapisuje plik.
        """
        # Tworzenie folderu 'dane15' i folderu dla danej litery
        base_folder = os.path.join(os.path.dirname(__file__), 'dane15')
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        # Tworzenie podfolderu dla litery
        if letter.islower():
        # Jeśli litera jest w lowercase, dodaj '.' przed nazwą folderu
            folder_name = f".{letter.strip()}"
        else:
            folder_name = letter.strip()

        # Jeśli folder istnieje, usuń go i stwórz ponownie
        letter_folder = os.path.join(base_folder, folder_name)  # Usunięcie zbędnych spacji w nazwie
        if os.path.exists(letter_folder):
            shutil.rmtree(letter_folder)

        os.makedirs(letter_folder)

        # Tworzenie obrazu na podstawie binarnej reprezentacji
        img = Image.new("1", (self.segment_size, self.segment_size), 1)  # "1" to tryb czarno-biały
        pixels = []

        for row in binary_matrix:
            for pixel in row:
                pixels.append(0 if pixel == 1 else 255)  # 0 = czarny, 255 = biały

        img.putdata(pixels)
        img = img.resize((self.segment_size * 10, self.segment_size * 10))  # Powiększ, by był bardziej widoczny

        # Zapisz obraz do pliku .png
        img.save(os.path.join(letter_folder, f"{letter}.png"))

    def display_binary_in_console(self, letter, binary_matrix):
        """
        Wyświetla mapę binarną w konsoli.
        """
        # Wyświetlanie binarnej reprezentacji w konsoli
        print(f"Binarna reprezentacja dla '{letter}':")
        for row in binary_matrix:
            print(" ".join(str(pixel) for pixel in row))
        print()

    def on_generate(self):
        """Obsługuje kliknięcie przycisku generowania obrazu."""
        input_text = self.text_input.get()
        if not input_text:
            messagebox.showwarning("Uwaga", "Pole tekstowe nie może być puste!")
            return

        # Czyszczenie poprzednich obrazów w widoku
        #for widget in self.image_frame.winfo_children():
        #    widget.destroy()

        # Generowanie obrazów dla każdego znaku
        for idx, letter in enumerate(input_text):
            if letter.isspace():  # Pomijamy znaki spacji
                continue
            img = self.create_segment_image(letter)
            img_tk = ImageTk.PhotoImage(img)
            #label = tk.Label(self.image_frame, image=img_tk)
            #label.image = img_tk  # Referencja, aby obraz nie został usunięty przez garbage collector
            #label.grid(row=idx // 50, column=idx % 50, padx=1, pady=1)  # 50 znaków na linię

            # Generowanie binarnej reprezentacji
            binary_matrix = self.get_binary_representation(img)

            # Wyświetlenie mapy binarnej w konsoli
            self.display_binary_in_console(letter, binary_matrix)

            # Zapisanie obrazu jako .png w folderze
            self.save_binary_image(letter, binary_matrix)
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
