import os
import shutil
from PIL import Image, ImageDraw, ImageFont

def generate_training_dataset():
    """Generuje zbiór danych treningowych dla liter i cyfr."""
    dataset_dir = "dane1"
    img_size = (24, 24)  # Rozmiar obrazu
    font_size = 14       # Rozmiar czcionki
    num_samples = 2    # Liczba obrazów na znak

    # Usuń folder, jeśli istnieje, i utwórz go od nowa
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    # Lista znaków do generacji (małe i wielkie litery, cyfry)
    characters = [chr(i) for i in range(ord('a'), ord('z') + 1)] + \
                 [chr(i).upper() for i in range(ord('a'), ord('z') + 1)] + \
                 [str(i) for i in range(10)]

    for char in characters:
        # Nazwa podfolderu
        folder_name = f".{char}" if char.islower() else char
        char_folder = os.path.join(dataset_dir, folder_name)
        os.makedirs(char_folder, exist_ok=True)

        for sample_idx in range(num_samples):
            # Tworzenie obrazu
            image = Image.new("RGB", img_size, color="white")
            draw = ImageDraw.Draw(image)

            # Ustawienie czcionki
            try:
                font_path = "courier.ttf"  # Ścieżka do czcionki
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                font = ImageFont.load_default()

            # Wyśrodkowanie tekstu
            bbox = font.getbbox(char)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_position = ((img_size[0] - text_width) // 2, (img_size[1] - text_height) // 2)

            draw.text(text_position, char, font=font, fill="black")

            # Zapisanie obrazu
            img_path = os.path.join(char_folder, f"{char}_{sample_idx}.png")
            image.save(img_path)

    print(f"Zbiór danych treningowych został wygenerowany w folderze '{dataset_dir}'.")
generate_training_dataset()