from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
from utils import preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sequences(Dataset):
    def __init__(self, path, max_seq_len):
        self.max_seq_len = max_seq_len
        df = pd.read_csv(path)
        vectorizer = CountVectorizer(max_df=0.95)
        vectorizer.fit(df.free_text.tolist())

        self.token2idx = vectorizer.vocabulary_
        self.token2idx['<PAD>'] = max(self.token2idx.values()) + 1

        tokenizer = vectorizer.build_analyzer()
        self.encode = lambda x: [self.token2idx[token] for token in tokenizer(x)
                                 if token in self.token2idx]
        self.pad = lambda x: x + (max_seq_len - len(x)) * [self.token2idx['<PAD>']]

        sequences = [self.encode(sequence)[:max_seq_len] for sequence in df.free_text.tolist()]
        sequences, self.labels = zip(*[(sequence, label) for sequence, label
                                    in zip(sequences, df.label_id.tolist()) if sequence])
        self.sequences = [self.pad(sequence) for sequence in sequences]

    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.labels[i]

    def __len__(self):
        return len(self.sequences)
    
class GRU_Classification(nn.Module):
    def __init__(
        self,
        vocab_size,
        batch_size,
        embedding_dimension=100,
        hidden_size=128,
        n_layers=1,
        device='cpu',
    ):
        super(GRU_Classification, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device
        self.batch_size = batch_size

        self.encoder = nn.Embedding(vocab_size, embedding_dimension)
        self.rnn = nn.GRU(
            embedding_dimension,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(hidden_size, 1)

    def init_hidden(self):
        h_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(self.device)
        # c_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device)
        return h_0

    def forward(self, inputs):
        # Avoid breaking if the last batch has a different size
        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        encoded = self.encoder(inputs)
        output, _ = self.rnn(encoded, self.init_hidden())
        output = self.decoder(output[:, :, -1]).squeeze()
        return output

train_csv= "./data/train.csv"
train_dataset = Sequences(train_csv, max_seq_len=128)
state_dict_path = "./model/gru_state_dict.pt"

best_model = GRU_Classification(
    hidden_size=128,
    vocab_size=len(train_dataset.token2idx),
    device=device,
    batch_size=1,
)
best_model.load_state_dict(torch.load(state_dict_path,map_location=torch.device(device)))
best_model.to(device)
best_model.eval()

def gru_sentiment_classification(text):
    with torch.no_grad():
        text = preprocess(text)
        test_vector = torch.LongTensor([train_dataset.pad(train_dataset.encode(text))]).to(device)
        test_vector = test_vector[:,0:128]
        output = best_model(test_vector)
        prediction = torch.sigmoid(output).item()
        list_label = ["CLEAN", 'HATE']
        if prediction > 0.5:
          label_pred = [prediction/100, (100 -prediction)/100]
        else:
          label_pred = [(100 - prediction)/100, prediction/100]
    result_dict = dict(zip(list_label, label_pred))
    return result_dict

print(gru_sentiment_classification('thằng chó này'))
