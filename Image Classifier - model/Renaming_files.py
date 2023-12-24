import os

def zmien_nazwy(folder_path, rozszerzenie):
    try:
        # Sprawdź czy folder istnieje
        if not os.path.exists(folder_path):
            print(f"Podany folder '{folder_path}' nie istnieje.")
            return

        licznik = 55

        # Przejdź przez wszystkie pliki w folderze
        for filename in os.listdir(folder_path):
            # Sprawdź czy to plik z odpowiednim rozszerzeniem
            if filename.endswith(rozszerzenie):
                # Generuj nową nazwę pliku
                nowa_nazwa = str(licznik)+'.jpg'

                # Pełna ścieżka do pliku przed zmianą
                stara_sciezka = os.path.join(folder_path, filename)

                # Pełna ścieżka do pliku po zmianie
                nowa_sciezka = os.path.join(folder_path, nowa_nazwa)

                # Zmiana nazwy pliku
                os.rename(stara_sciezka, nowa_sciezka)

                licznik += 1

                #print(f"Zmieniono nazwę pliku: {filename} na {nowa_nazwa}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")

folder_do_zmiany = ""
rozszerzenie_plikow = ".jpg"

zmien_nazwy(folder_do_zmiany, rozszerzenie_plikow)