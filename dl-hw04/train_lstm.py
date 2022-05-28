from typing import Callable
import numpy as np
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
import torchtext.vocab as tvcb
from torchtext.datasets import SST2
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence


EMB_DIM = 300
HIDDEN_SIZE = 300
FC_SIZE = 4 * HIDDEN_SIZE
BATCH_SIZE = 256
EPOCHS = 50


def main():
    tokenizer = get_tokenizer('basic_english')
    embeddings_full = tvcb.GloVe('840B', dim=EMB_DIM)
    train_dataset = SST2(split='train')
    vocab, embeddings = get_vocab_embeddings(train_dataset, tokenizer,
                                             embeddings_full)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataset = SST2(split='dev')
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm = LSTM(tokenizer, vocab, embeddings, device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lstm.parameters())
    lstm.train()

    for i in range(30):
        cur_loss = 0.0
        hits = 0
        total = 0
        for texts, labels in train_dataloader:
            X, y, lengths = lstm.texts_to_tensor(texts, labels)
            logits = lstm(X, lengths)
            predictions = torch.round(torch.sigmoid(logits.detach())).int()
            hits += (predictions == y).sum().item()
            total += len(y)
            loss = loss_function(logits, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cur_loss += loss.item()
        print(f'Epoch {i + 1}: loss = {cur_loss:.2f}')
        print(f'Train accuracy = {hits * 100 / total:.1f}%')

        lstm.eval()
        hits = 0
        total = 0
        with torch.no_grad():
            for texts, labels in val_dataloader:
                X, y, lengths = lstm.texts_to_tensor(texts, labels)
                logits = lstm(X, lengths)
                predictions = torch.round(torch.sigmoid(logits)).int()
                hits += (predictions == y).sum().item()
                total += len(y)
        print(f'Validation accuracy {(100 * hits / total):.1f}%\n')
        lstm.train()


class LSTM(nn.Module):
    def __init__(self, tokenizer: Callable, vocab: tvcb.Vocab,
                 embeddings: np.ndarray, device: torch.device):
        super().__init__()
        self.input_size = embeddings.shape[1]
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embeddings, dtype=torch.float32),
            freeze=True
        ).to(device)
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
            bidirectional=False
        ).to(device)
        self.stack = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, FC_SIZE),
            nn.ReLU(),
            nn.Linear(FC_SIZE, 1)
        ).to(device)

    def forward(self, X: torch.tensor, lengths: torch.tensor):
        X = self.embedding(X)
        X = pack_padded_sequence(X, lengths, batch_first=True,
                                 enforce_sorted=False)
        _, (h, _) = self.rnn(X)
        logits = self.stack(h)
        return logits.squeeze()

    def texts_to_tensor(self, texts: tuple[str], labels: torch.tensor):
        tokens = [self.vocab(self.tokenizer(text)) for text in texts]
        lengths = torch.tensor([len(token_list) for token_list in tokens],
                               dtype=torch.int64)
        lengths, perm_inds = lengths.sort(0, descending=True)
        max_length = lengths[0].item()
        tokens_arr = np.zeros((len(tokens), max_length), dtype='int64')
        for i, ind in enumerate(perm_inds):
            j = len(tokens[ind])
            tokens_arr[i, :j] = np.asarray(tokens[ind])
        X = torch.tensor(tokens_arr, device=self.device)
        y = labels[perm_inds].to(self.device)
        return X, y, lengths


def get_vocab_embeddings(dataset: Dataset,
                         tokenizer: Callable,
                         vectors: tvcb.Vectors
                         ) -> tuple[tvcb.Vocab, np.ndarray]:
    def yield_tokens(data_iter):
        for text, _ in data_iter:
            yield tokenizer(text)

    vocab = tvcb.build_vocab_from_iterator(yield_tokens(dataset),
                                           specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    embeddings = np.zeros((len(vocab), EMB_DIM))
    for i, word in enumerate(vocab.get_itos()):
        embeddings[i] = vectors[word]

    return vocab, embeddings


if __name__ == '__main__':
    main()
