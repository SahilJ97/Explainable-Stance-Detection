from abc import ABC
import torch.nn as nn
import torch
import transformers


class VastClassifier(nn.Module, ABC):
    num_labels = 3

    def __init__(self, doc_len, pretrained_model="bert-base-uncased"):
        super().__init__()
        self.doc_len = doc_len
        self.bert_model = transformers.BertModel.from_pretrained(
            pretrained_model,
        )

    def to(self, *args, **kwargs):
        self.bert_model = self.bert_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def extract_co_embeddings(self, pad_mask, doc_stopword_mask, inputs=None, inputs_embeds=None,
                              token_type_ids=None, return_attentions=False):
        # Inputs/inputs_embeds are organized as "[CLS] document [SEP] topic [SEP]"
        if inputs is None and inputs_embeds is None:
            raise ValueError("Either inputs or inputs_embeds must be provided")
        if inputs is not None:
            inputs_embeds = self.get_inputs_embeds(inputs)
        if hasattr(self, "fix_bert") and self.fix_bert:
            with torch.no_grad():
                last_hidden_state, pooler_outputs, attentions = self.bert_model.forward(
                    inputs_embeds=inputs_embeds,
                    attention_mask=pad_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True
                )
        else:
            last_hidden_state, pooler_outputs, attentions = self.bert_model.forward(  # gets stuck here! WHY?
                inputs_embeds=inputs_embeds,
                attention_mask=pad_mask,
                token_type_ids=token_type_ids,
                output_attentions=True
            )
        doc_token_counts = torch.sum(
            pad_mask[:, 1:self.doc_len + 1:] * doc_stopword_mask,
            dim=-1
        )
        topic_token_counts = torch.sum(
            pad_mask[:, self.doc_len + 2:-1],
            dim=-1
        )
        last_hidden_state = torch.unsqueeze(pad_mask, dim=-1) * last_hidden_state
        doc_embeds = last_hidden_state[:, 1:self.doc_len + 1:]
        doc_embeds = torch.unsqueeze(doc_stopword_mask, dim=-1) * doc_embeds
        topic_embeds = last_hidden_state[:, self.doc_len + 2:-1]
        doc = torch.sum(doc_embeds, dim=1) / doc_token_counts[:, None]  # same idea for document embeddings
        topic = torch.sum(topic_embeds, dim=1) / topic_token_counts[:, None]  # avg of non-zeroed topic tokens
        return_list = [doc, topic]
        if return_attentions:
            return_list.append(attentions)
        return tuple(return_list)

    def get_inputs_embeds(self, inputs):
        inputs = inputs.long()
        token_type_ids = torch.zeros_like(inputs, dtype=torch.long)
        return self.bert_model.embeddings(input_ids=inputs, token_type_ids=token_type_ids)
