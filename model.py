import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.fc_query = nn.Linear(hidden_dim, hidden_dim)
        self.fc_key = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        # Linear transformations
        Q = self.fc_query(query)
        K = self.fc_key(key)
        V = self.fc_value(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(query.device)
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Concatenate heads and apply final linear layer
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.fc_out(context)

        return output, attn_weights

class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(hidden_dim, num_heads)

    def forward(self, encoder_outputs, decoder_hidden):
        query = decoder_hidden.unsqueeze(1)
        context, _ = self.attention(query, encoder_outputs, encoder_outputs)
        context = context.squeeze(1)
        return context

class RMSNorm(nn.Module):
  def __init__(self, hidden_dim):
    super(RMSNorm, self).__init__()
    self.hidden_dim = hidden_dim
    self.layer_norm = nn.LayerNorm(hidden_dim)

  def forward(self, x):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
    x = x / (rms + 1e-5) # Küçük bir değer ekliyerek sıfıra bölme hatasını önler
    x = self.layer_norm(x)
    return x


class DilModeli(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout_prob=0.5, num_heads=4):
        super(DilModeli, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.attention = Attention(hidden_dim, num_heads)

        self.rms_norm = RMSNorm(hidden_dim)

        # Feedforward Katmanı
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),  
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Normalize Katmanı
        self.normalize = nn.LayerNorm(512)  

        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        rnn_out, (hidden, cell) = self.rnn(x)

        rnn_out = self.rms_norm(rnn_out)

        rnn_out_with_dim = rnn_out.unsqueeze(1)

        context = self.attention(rnn_out_with_dim, rnn_out_with_dim[:, -1, :])

        rnn_out = rnn_out.unsqueeze(1)
        combined_out = torch.cat([context.unsqueeze(1), rnn_out], dim=-1)

        # Feedforward Katmanı
        combined_out = self.feedforward(combined_out.view(-1, combined_out.size(-1)))

        # Normalize Katmanı
        combined_out = self.normalize(combined_out)

        out = self.fc(combined_out)
        return out


