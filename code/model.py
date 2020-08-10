import torch
import torch.nn as nn
import torch.nn.functional as F

# Example from pytorch website
class TextSentiment(nn.Module):
    def __init__(
        self, vocab_size, vocab, embed_dim, num_class, num_hidden=50, freeze=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.EmbeddingBag.from_pretrained(vocab.vectors, freeze=freeze)
        self.fc1 = nn.Linear(embed_dim + 1, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_class)

    def forward(self, text, text_len):
        embedded = self.embedding(text)
        embedded = embedded[:, : self.embed_dim]
        embedded = torch.cat(
            [
                embedded,
                torch.tensor(
                    torch.reshape(text_len, (text_len.shape[0], 1)), dtype=torch.float
                ),
            ],
            dim=1,
        )
        hidden = F.relu(self.fc1(embedded))
        hidden = F.relu(self.fc2(hidden))
        return self.fc3(hidden)


class DensityRatioModel(nn.Module):
    def __init__(
        self, vocab_size, vocab, embed_dim, num_class, num_hidden=50, freeze=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.EmbeddingBag.from_pretrained(vocab.vectors, freeze=freeze)
        self.fc1 = nn.Linear(embed_dim + 1, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_class)

    def forward(self, text, text_len):
        embedded = self.embedding(text)
        embedded = embedded[:, : self.embed_dim]
        embedded = torch.cat(
            [
                embedded,
                torch.tensor(
                    torch.reshape(text_len, (text_len.shape[0], 1)), dtype=torch.float
                ),
            ],
            dim=1,
        )
        hidden = F.relu(self.fc1(embedded))
        return self.fc2(hidden)


class TextYearModel(nn.Module):
    def __init__(
        self, vocab_size, vocab, embed_dim, num_class, num_hidden=50, freeze=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.EmbeddingBag.from_pretrained(vocab.vectors, freeze=freeze)
        self.fc1 = nn.Linear(embed_dim + 2, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_class)

    def forward(self, text, text_len, year):
        embedded = self.embedding(text)
        embedded = embedded[:, : self.embed_dim]
        embedded = torch.cat(
            [
                embedded,
                torch.tensor(
                    torch.reshape(text_len, (text_len.shape[0], 1)), dtype=torch.float
                ),
                torch.tensor(
                    torch.reshape(year, (year.shape[0], 1)), dtype=torch.float
                ),
            ],
            dim=1,
        )
        hidden = F.relu(self.fc1(embedded))
        hidden = F.relu(self.fc2(hidden))
        return self.fc3(hidden)
