from torch.utils.data import Dataset
from transformers.tokenization_bert import BertTokenizer
import tokenizations
from sys import stderr
import torch
from nltk.corpus import stopwords
import string
from csv import DictReader

sw = stopwords.words("english")
punc = [c for c in string.punctuation]


def contains_alpha(s):
    for c in s:
        if c.isalpha():
            return True
    return False


def crop_or_pad(seq, length, padding_item="[PAD]"):
    # Cropping/padding occurs on right hand side
    seq = list(seq)
    if len(seq) > length:
        seq = seq[:length]
    else:
        seq.extend([padding_item for i in range(length - len(seq))])
    return seq


def get_stopword_mask(s):
    if type(s) == str:
        s = s.split()
    mask = []
    for word in s:
        if word in sw:
            mask.append(0.)
        else:
            mask.append(1.)
    return mask


class VastReader(Dataset):
    doc_len = 250
    topic_len = 10
    max_len = topic_len + doc_len + 3

    def __init__(self,
                 main_csv,
                 exclude_from_main=None,
                 word_importance_csv=None,
                 smoothing=None,
                 smooth_param=.01,
                 relevance_type="binary",
                 tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
                 use_tf_idf_as_labels=False,
                 return_seen=False,
                 no_stopwords=False
                 ):
        """
        :param main_csv: Path to data CSV file
        :param exclude_from_main: Path to file containing new_id values of datapoints that should NOT be loaded from
        main_csv. If word_importance_csv is defined, all of its datapoints should be represented in this list
        :param word_importance_csv: Path to CSV file for data with word importance annotations. Word importance scores
        should be
        """
        self.main_csv = main_csv
        self.exclude_from_main = []
        if exclude_from_main:
            with open(exclude_from_main, "r") as f:
                for line in f:
                    self.exclude_from_main.append(line.strip())
        self.word_importance_csv = word_importance_csv
        self.inputs, self.labels = [], []
        if smoothing not in [None, "simple", "tf-idf"]:
            print("Error: smoothing must be one of: ['simple', 'tf-idf']", file=stderr)
            exit(1)
        self.smoothing = smoothing
        self.smooth_param = smooth_param
        self.relevance_type = relevance_type
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.use_tf_idf_as_labels = use_tf_idf_as_labels
        self.return_seen = return_seen
        self.no_stopwords = no_stopwords
        self.seen_dict = {}

        # Count topics (for TF-IDF computations)
        all_topics = set()
        with open(main_csv, "r") as f:
            reader = DictReader(f)
            for row in reader:
                all_topics.add(row["new_topic"])
        self.n_topics = len(all_topics)

        self.load_data()

    def new_id_to_seen(self, new_id):
        return int(self.seen_dict[str(new_id)])

    def new_token_mapping(self, text, orig_tokens, orig_value_mapping):
        new_tokens = self.tokenizer.tokenize(text)
        new2old, old2new = tokenizations.get_alignments(new_tokens, orig_tokens)
        new_token_scores = []
        for i in range(len(new_tokens)):
            if len(new2old[i]) > 0:
                new_token_scores.append(
                    max([orig_value_mapping[j] for j in new2old[i]])
                )
            else:
                new_token_scores.append(0.)
        return new_tokens, new_token_scores

    def relevance_scores(self, orig_tokens, tf_idfs=None):
        if self.relevance_type == "tf-idf":
            scores = list(tf_idfs)
        elif self.relevance_type == "binary":
            scores = [1 for i in range(len(orig_tokens))]
        else:
            return None
        for i in range(len(orig_tokens)):
            if not contains_alpha(orig_tokens[i]):
                scores[i] = 0
        return scores

    def smooth(self, orig_value_mapping, tf_idfs=None):
        """Preserves the original token-value mapping (spacy tokens, rather than transformer tokens)"""
        new_values = []
        n = len(orig_value_mapping)
        if self.smoothing == "tf-idf":
            denom = sum(orig_value_mapping)
            for tf_idf in tf_idfs:
                denom += self.smooth_param * tf_idf
            for v, tf_idf in zip(orig_value_mapping, tf_idfs):
                new_values.append(
                    (v + self.smooth_param * tf_idf) / denom
                )
        elif self.smoothing == "simple":
            denom = sum(orig_value_mapping) + (n * self.smooth_param)
            for v in orig_value_mapping:
                new_values.append(
                    (v + self.smooth_param) / denom
                )
        elif not self.smoothing:
            denom = sum(orig_value_mapping)
            for v in orig_value_mapping:
                new_values.append(v / denom)
        return new_values

    def load_data(self):
        CLS_ID = self.tokenizer.tokenize("[CLS]")
        SEP_ID = self.tokenizer.tokenize("[SEP]")
        with open(self.main_csv, "r") as f:
            reader = DictReader(f)
            for row in reader:
                self.seen_dict[row["new_id"]] = row["seen?"]
                if row["new_id"] in self.exclude_from_main:
                    continue
                self.labels.append(int(row["label"]))
                if not self.no_stopwords:
                    doc_tokens = self.tokenizer.tokenize(row["post"])
                else:
                    doc_tokens = self.tokenizer.tokenize(row["text_s"])
                doc_tokens = crop_or_pad(doc_tokens, self.doc_len)
                doc_stopword_mask = get_stopword_mask(doc_tokens)
                doc_tokens = CLS_ID + doc_tokens + SEP_ID
                topic_tokens = self.tokenizer.tokenize(row["topic_str"])
                topic_tokens = crop_or_pad(topic_tokens, self.topic_len)
                topic_tokens = topic_tokens + SEP_ID
                input_dict = {
                    "input_tokens": doc_tokens + topic_tokens,
                    "doc_stopword_mask": doc_stopword_mask,
                    "weights": None,
                    "relevance_scores": None,
                    "new_id": int(row["new_id"])
                }
                self.inputs.append(input_dict)

        if self.word_importance_csv is None:
            return
        try:
            with open(self.word_importance_csv, "r") as f:
                reader = DictReader(f)
                for row in reader:
                    self.labels.append(int(row["label"]))
                    topic_tokens = self.tokenizer.tokenize(" ".join([t for t in row["topic"].split(" ") if t not in sw]))
                    topic_tokens = crop_or_pad(topic_tokens, self.topic_len)
                    topic_tokens = topic_tokens + SEP_ID
                    orig_word_weight_tuples = eval(row["weights"])
                    argument = row["argument"]
                    if self.no_stopwords:
                        argument = " ".join([t for t in argument.split(" ") if t not in sw])
                    orig_tokens, orig_weight_mapping = zip(*orig_word_weight_tuples)
                    orig_tf_idf_tuples = eval(row["tf-idf"])
                    _, orig_tf_idfs = zip(*orig_tf_idf_tuples)
                    if self.use_tf_idf_as_labels:
                        orig_weight_mapping = self.smooth(orig_tf_idfs, orig_tf_idfs)
                    else:
                        orig_weight_mapping = self.smooth(orig_weight_mapping, orig_tf_idfs)
                    doc_tokens, weights = self.new_token_mapping(argument, orig_tokens, orig_weight_mapping)
                    doc_tokens = crop_or_pad(doc_tokens, self.doc_len)
                    doc_stopword_mask = get_stopword_mask(doc_tokens)
                    weights = crop_or_pad(weights, self.doc_len, padding_item=0.)
                    doc_tokens = CLS_ID + doc_tokens + SEP_ID
                    weights = [0] + weights + [0]
                    weight_sum = sum(weights)
                    weights = [w / weight_sum for w in weights]  # re-normalize (after potentially cropping)
                    relevance_scores = self.relevance_scores(orig_tokens, orig_tf_idfs)
                    _, relevance_scores = self.new_token_mapping(
                        argument,
                        orig_tokens,
                        relevance_scores
                    )
                    relevance_scores = crop_or_pad(relevance_scores, self.doc_len, padding_item=0.)
                    relevance_scores = [0] + relevance_scores + [0]
                    input_dict = {
                        "input_tokens": doc_tokens + topic_tokens,
                        "doc_stopword_mask": doc_stopword_mask,
                        #"topic_stopword_mask": topic_stopword_mask,
                        "weights": weights,
                        "relevance_scores": relevance_scores,
                        "new_id": int(row["new_id"])
                    }
                    self.inputs.append(input_dict)
        except TypeError:  # Handle weird error that occurs AFTER the entire file is read
            pass

    def __getitem__(self, idx):
        ip = self.inputs[idx]
        if ip["weights"]:
            has_attribution_label = True
            weights = ip["weights"]
            relevance_scores = ip["relevance_scores"]
        else:
            has_attribution_label = False
            weights = [0. for _ in range(2 + self.doc_len)]
            relevance_scores = weights
        input_seq = self.tokenizer.convert_tokens_to_ids(
            ip["input_tokens"]
        )
        attribution_info = (has_attribution_label, torch.tensor(weights), torch.tensor(relevance_scores))
        return_vals = [
            torch.tensor(input_seq), self.labels[idx], torch.tensor(ip["doc_stopword_mask"]), attribution_info
        ]
        if self.return_seen:
            return_vals.append(self.new_id_to_seen(ip["new_id"]))
        return tuple(return_vals)

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    dataset = VastReader(
        "../data/VAST/vast_train_0.9.csv",
        exclude_from_main="../data/VAST_word_importance/special_datapoints_train.txt",
        word_importance_csv="../data/VAST_word_importance/processed_annotated_train.csv",
        relevance_type="tf-idf"
    )
    print(len(dataset))
