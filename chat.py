import torch
import torch.nn.functional as F
from model import DilModeli
from data import DataLoader

loader = DataLoader("/tokenler_path","/kelime_token_path")

loader.load_word_dict()
if loader.load_word_dict:
    loader.load_data()
    X, y, kelime_sozluk, = loader.get_data()


# Modeli oluştur
vocab_size = len(kelime_sozluk)
embedding_dim = 60
hidden_dim = 128
num_layers = 1
dropout_prob = 0.2
num_heads = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DilModeli(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob, num_heads).to(device)

# Modelin yüklenmesi
try:

  checkpoint = torch.load('/dil_modeli_multihead_attention_checkpoint.pth')

  model.load_state_dict(checkpoint['model_state_dict'])
except FileNotFoundError:
    print("Hata: Model dosyası bulunamadı.")


model.train()



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-1e9):
    assert logits.dim() == 1 or logits.dim() == 2

    if logits.dim() == 2:
        logits = logits[-1]

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value


    return logits

def sample_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    probabilities = F.softmax(logits, dim=-1)
    sampled_index = torch.multinomial(probabilities, 1)
    return sampled_index.item()

def generate_text(model, seed_words, length=10, temperature=1.0, top_k=0, top_p=0.0):
    with torch.no_grad():
        current_words = seed_words
        generated_text = current_words.copy()

        for _ in range(length):
            current_indices = [kelime_sozluk.get(word, -1) for word in current_words]
            current_indices = [idx for idx in current_indices if idx != -1]  # Modelde bulunmayan kelimeleri atla
            if not current_indices:
                break  # Modelde bulunan kelimeler olmadığı için dur
            input_tensor = torch.tensor(current_indices, dtype=torch.long)
            output = model(input_tensor)

            filtered_logits = top_k_top_p_filtering(output[-1], top_k=top_k, top_p=top_p)
            next_index = sample_with_temperature(filtered_logits, temperature)
            next_word = list(kelime_sozluk.keys())[list(kelime_sozluk.values()).index(next_index)]

            generated_text.append(next_word)
            current_words = generated_text[-len(seed_words):]

    return ' '.join(generated_text)

def interact_with_model(model, kelime_sozluk, seed_words, max_length=100, temperature=0.2, top_k=50, top_p=0.9):
    current_words = seed_words
    generated_text = current_words.copy()

    for _ in range(max_length):
        current_indices = [kelime_sozluk.get(word, -1) for word in current_words]
        current_indices = [idx for idx in current_indices if idx != -1]  # Modelde bulunmayan kelimeleri atla
        if not current_indices:
            break  # Modelde bulunan kelimeler olmadığı için dur
        input_tensor = torch.tensor(current_indices, dtype=torch.long)

        output = model(input_tensor)

        filtered_logits = top_k_top_p_filtering(output[-1], top_k=top_k, top_p=top_p)
        next_index = sample_with_temperature(filtered_logits, temperature)
        next_word = list(kelime_sozluk.keys())[list(kelime_sozluk.values()).index(next_index)]

        generated_text.append(next_word)
        current_words = generated_text[-len(seed_words):]

    return ' '.join(generated_text)

# Kullanıcıdan başlangıç kelimelerini al
seed_words = input("Machine-ai: ").split()



# Her bir metni oluştur ve yazdır
for i in range(1):
    generated_text = interact_with_model(model, kelime_sozluk, seed_words, max_length=100, temperature=0.2, top_k=50, top_p=0.9)
    words = generated_text.split()
    formatted_text = ''
    for j, word in enumerate(words):
        if (j + 1) % 15 == 0:
            formatted_text += word + '\n\n'
        else:
            formatted_text += word + ' '
    print("Machine Tarafından Oluşturulan metin", i+1, ":\n", formatted_text)
