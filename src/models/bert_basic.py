from abc import ABC
import torch
from transformers import BertForSequenceClassification
from src.models.vast_classifier import VastClassifier


class BertBasic(VastClassifier, ABC):
    loss = torch.nn.CrossEntropyLoss()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bert_clf = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    def to(self, *args, **kwargs):
        self.bert_clf.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, return_attentions=False, **kwargs):
        for k in ["token_type_ids", "inputs", "inputs_embeds"]:
            if k not in kwargs.keys():
                kwargs[k] = None
        loss, logits, attentions = self.bert_clf(
            input_ids=kwargs["inputs"],
            inputs_embeds=kwargs["inputs_embeds"],
            token_type_ids=kwargs["token_type_ids"],
            attention_mask=kwargs["pad_mask"],
            labels=kwargs["labels"],
            output_attentions=True
        )
        return_list = [logits]
        if "labels" in kwargs.keys():
            return_list.append(loss)
        if return_attentions:
            return_list.append(attentions)
        return tuple(return_list)
