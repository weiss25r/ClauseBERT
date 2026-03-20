import torch.nn as nn

class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, num_classes)

    def forward(self, encoding):
        outputs = self.bert(input_ids=encoding['input_ids'],attention_mask=encoding['attention_mask'])
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        logits = self.linear(cls_output)
        return logits