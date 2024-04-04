import pickle

class DataLoader:
    def __init__(self, word_dict_path, data_path):
        self.word_dict_path = word_dict_path
        self.data_path = data_path
        self.word_dict = None
        self.X = None
        self.y = None
        self.model = None

    def load_word_dict(self):
        try:
            with open(self.word_dict_path, "rb") as f:
                self.word_dict = pickle.load(f)
        except FileNotFoundError:
            print("Hata: Kelime sözlüğü bulunamadı.")
    
    def load_data(self):
        try:
            with open(self.data_path, "rb") as f:
                self.X, self.y = pickle.load(f)
            print("Veriler ve tokenler başarıyla yüklendi.")
        except FileNotFoundError:
            print("Hata: Veri dosyası bulunamadı.")
    
  
    def get_data(self):
        return self.X, self.y, self.word_dict
