from abc import ABC
import torch
import torch.nn as nn
from src.models.vast_classifier import VastClassifier


class BertJoint(VastClassifier, ABC):
    loss = torch.nn.CrossEntropyLoss()

    def __init__(self, fix_bert=False, **kwargs):
        super(BertJoint, self).__init__(**kwargs)
        self.fix_bert = fix_bert
        self.dropout = nn.Dropout(p=.2046)
        self.hidden_layer = torch.nn.Linear(768*2, 283, bias=False)
        self.output_layer = torch.nn.Linear(283, 3)

    def to(self, *args, **kwargs):
        self.hidden_layer.to(*args, **kwargs)
        self.output_layer.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, **kwargs):
        for k in ["token_type_ids", "inputs", "inputs_embeds"]:
            if k not in kwargs.keys():
                kwargs[k] = None
        return_attentions = "return_attentions" in kwargs.keys() and kwargs["return_attentions"]
        co_embeds = self.extract_co_embeddings(
            pad_mask=kwargs["pad_mask"],
            doc_stopword_mask=kwargs["doc_stopword_mask"],
            inputs=kwargs["inputs"],
            inputs_embeds=kwargs["inputs_embeds"],
            token_type_ids=kwargs["token_type_ids"],
            return_attentions=return_attentions
        )
        if not return_attentions:
            doc, topic = co_embeds
        else:
            doc, topic, attentions = co_embeds
        both_embeds = torch.cat([doc, topic], dim=-1)
        if "use_dropout" in kwargs.keys() and kwargs["use_dropout"]:
            both_embeds = self.dropout(both_embeds)
        hl = self.hidden_layer(both_embeds)
        hl = torch.nn.functional.tanh(hl)
        ol = self.output_layer(hl)
        return_list = [ol]
        if "labels" in kwargs.keys():
            return_list.append(self.loss(ol, kwargs["labels"]))
        if return_attentions:
            return_list.append(attentions)
        return tuple(return_list)
