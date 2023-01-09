import torch
from torch import nn
from transformers import DistilBertModel


class FakeDetectorModel(nn.Module):
    def __init__(self, bert):
        super(FakeDetectorModel, self).__init__()
        self.bert = bert
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        #self.fc1 = nn.Linear(512, 512)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, input_id, mask):
        x = self.bert(input_ids=input_id, attention_mask=mask)
        x = x[0]
        x = x[:, 0]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        y_pred = torch.sigmoid(x)
        return y_pred
