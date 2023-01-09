import pandas
from torch.utils.data import Dataset as DS


class Dataset(DS):
    def __init__(self, df: pandas.DataFrame):
        self.labels = df["label"].tolist()
        self.tokens = df["tokens"].tolist()
        self.attention_masks = df["attention_masks"].tolist()

    def get_labels(self):
        return self.labels

    def get_tokens(self):
        return self.tokens

    def get_attention_masks(self):
        return self.attention_masks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx], self.attention_masks[idx]

