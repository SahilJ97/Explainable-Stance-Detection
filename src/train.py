import torch
from src.vast_reader import VastReader
from src.models import BertJoint, BertBasic
from src.utils import get_pad_mask
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.optim import Adam
from sklearn.metrics import f1_score
from src import visualize
import numpy as np
import argparse
import sys
import gc

# Parse arguments
true_strings = ['t', 'true', '1', 'yes', 'y', ]
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--relevance_type', help='Type of relevance scores ("binary" or "tf-idf")',
                    required=False, default="binary")
parser.add_argument('-p', '--prior',
                    help='Attribution prior: "ig" for Integrated Gradients, "att" for BERT attention, or "none". '
                         'Must be "none" if model is *not* bert-joint-new',
                    required=True)
parser.add_argument('-b', '--batch_size', type=int, required=True)
parser.add_argument('-l', '--learning_rate', type=float, required=True)
parser.add_argument('--lambda', help='Prior loss coefficient', type=float, required=False)
parser.add_argument('-o', '--output_name',
                    help='Base filename (without suffix) to which the best model will be saved in ~/output/',
                    required=True)
parser.add_argument('-m', '--model_type', help='bert-joint-new, bert-joint, bert-basic', required=True)  # TODO
parser.add_argument('-s', '--random_seed', type=int, required=True)
parser.add_argument('--main_csv', required=False, default='../data/VAST/vast_train.csv')
parser.add_argument('-e', '--num_epochs', type=int, default=20)
parser.add_argument('--grad_steps', help='Number of batches per instance of backpropagation', type=int,
                    default=1)
parser.add_argument('--ig_steps', help='Number of Riemann steps used to compute Integrated Gradients. '
                                       'GI is equivalent to IG with 1 step', type=int,
                    default=1)
parser.add_argument('--bert_layers', help='Number of BERT attention layers to use for prior (if "prior" set to "att"). '
                                          'The LAST n layers will be used.', type=int, default=12)
parser.add_argument('--do_eval', dest='do_eval', action='store_true',
                    help='Perform evaluation on trained model')
parser.set_defaults(do_eval=False)
parser.add_argument('--use_tfidf_as_labels', dest='use_tfidf_as_labels', action='store_true',
                    help='Perform evaluation on trained model')
parser.set_defaults(use_tfidf_as_labels=False)
parser.add_argument('--load_weights', help='Load weights from what would be the output .pt file', dest='load_weights',
                    action='store_true', required=False)
parser.set_defaults(load_weights=False)
parser.add_argument('--generate_untuned', help='Create an untuned model, save, then exit', dest='generate_untuned',
                    action='store_true', required=False)
parser.set_defaults(generate_untuned=False)

args = vars(parser.parse_args())
relevance_type = args['relevance_type']
prior = args['prior']
batch_size = args['batch_size']
learning_rate = args['learning_rate']
lda = args['lambda']
output_name = args['output_name']
model_type = args['model_type']
seed = args['random_seed']
main_csv = args['main_csv']
num_epochs = args['num_epochs']
ig_steps = args['ig_steps']
grad_steps = args['grad_steps']
bert_layers = args['bert_layers']
do_eval = args['do_eval']
load_weights = args['load_weights']
use_tfidf_as_labels = args['use_tfidf_as_labels']
generate_untuned = args['generate_untuned']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def empty_cache():
    if DEVICE == "cuda":
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


def grad_sum_for_debug(module):
    if isinstance(module, DataParallel):
        module = module.module
    s = torch.as_tensor(0., dtype=torch.float)
    none_count = 0
    for p in module.bert_model.parameters():
        if p.grad is not None:
            s = s + torch.sum(p.grad)
        else:
            none_count += 1
    return s, none_count


def remove_all_grads():
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            obj.grad = None


def evaluate():
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model.eval()
    for html_file in [f"../output/{output_name}-eval-ig.html", f"../output/{output_name}-eval-att.html",
                      f"../output/{output_name}-eval-att-extra.html"]:
        with open(html_file, "w") as out_file:
            out_file.write(visualize.header)
    for file in [f"../output/{output_name}-eval-att.consistency", f"../output/{output_name}-eval-att.faithfulness",
                 f"../output/{output_name}-eval-rand.faithfulness", f"../output/{output_name}-eval-gxi.faithfulness"]:
        with open(file, "w") as out_file:
            out_file.write("")
    with open(f"../output/{output_name}-predictions.txt", "w") as out_file:
        out_file.write("")
    with open(f"../output/{output_name}-attributions.txt", "w") as out_file:
        out_file.write("")

    sum_ig_loss = 0.
    sum_ig_loss_seen = 0.
    sum_ig_loss_unseen = 0.
    sum_attention_loss = 0.
    sum_attention_loss_seen = 0.
    sum_attention_loss_unseen = 0.
    all_output_indices = []
    all_labels = []
    all_seen = []
    annotated_seen = []
    for i, data in enumerate(test_loader, 0):
        inputs, labels, doc_stopword_mask, attribution_info, seen = data
        all_seen.extend(seen)
        has_att_labels, weights, relevance_scores = attribution_info
        pad_mask = get_pad_mask(inputs).to(DEVICE)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        doc_stopword_mask = doc_stopword_mask.to(DEVICE)
        with torch.no_grad():
            outputs, correctness_loss, attentions = model.forward(
                pad_mask=pad_mask,
                doc_stopword_mask=doc_stopword_mask,
                inputs=inputs,
                use_dropout=False,
                token_type_ids=token_type_ids[:len(inputs)],
                labels=labels,
                return_attentions=True
            )
        output_indices = torch.max(outputs, dim=-1)[1]
        all_output_indices.extend(output_indices.tolist())
        all_labels.extend(labels.tolist())
        attentions = torch.stack(attentions)  # shape=(n_layers, batch_size, n_heads, seq_len, seq_len)

        for j in range(len(inputs)):
            if has_att_labels[j]:
                annotated_seen.append(seen[j])
                remove_all_grads()
                ig_attributions = integrated_gradients(
                    inputs[j],
                    labels[j],
                    x_pad_mask=pad_mask[j],
                    doc_stopword_mask=doc_stopword_mask[j],
                    allow_backprop=False
                )
                att_attributions = torch.sum(
                    attentions[-bert_layers:, j, :, :, :],
                    dim=(0, 1, 2)
                )
                for attribution_type, attributions in [("ig", ig_attributions), ("att", att_attributions)]:
                    abs_attributions = torch.abs(attributions)[:2 + test_set.doc_len]
                    attribution_scores = abs_attributions / torch.sum(abs_attributions)
                    weight_tensor, relevance_tensor = weights[j].to(DEVICE), relevance_scores[j].to(DEVICE)
                    prior_loss = lda * \
                        sum((weight_tensor - attribution_scores) ** 2 * relevance_tensor) / sum(relevance_tensor)
                    print(attribution_type, prior_loss)
                    if attribution_type == "ig":
                        sum_ig_loss += prior_loss
                        if seen[j] == 0:
                            sum_ig_loss_unseen += prior_loss
                        else:
                            sum_ig_loss_seen += prior_loss
                    else:
                        sum_attention_loss += prior_loss
                        if seen[j] == 0:
                            sum_attention_loss_unseen += prior_loss
                        else:
                            sum_attention_loss_seen += prior_loss

                    # Output visualization to file
                    tokens = test_set.tokenizer.convert_ids_to_tokens(inputs[j])
                    att_word_weights = abs_attributions * relevance_tensor
                    gold_word_weights = weight_tensor * relevance_tensor
                    attributions_html = visualize.get_words_html(
                        tokens,
                        att_word_weights.tolist()
                    )
                    weights_html = visualize.get_words_html(
                        tokens,
                        gold_word_weights.tolist()
                    )
                    with open(f"../output/{output_name}-eval-{attribution_type}.html", "a") as out_file:
                        out_file.write(f"<p>Model attributions:</p>\n{attributions_html}\n")
                        out_file.write(f"<p>Attribution labels:</p>\n{weights_html}\n")
                        out_file.write(f"<p>predicted, actual: {outputs[j].tolist(), labels[j].tolist()}</p>\n")
    print(f"Attribution loss using Integrated Gradients: {sum_ig_loss / len(test_loader) / batch_size}")
    print(f"For seen and unseen, respectively: "
          f"{sum_ig_loss_seen / annotated_seen.count(1)}, {sum_ig_loss_unseen / annotated_seen.count(0)}")
    print(f"Attribution loss using BERT Attention: {sum_attention_loss / len(test_loader) / batch_size}")
    print(f"For seen and unseen, respectively: "
          f"{sum_attention_loss_seen / annotated_seen.count(1)}, {sum_attention_loss_unseen / annotated_seen.count(0)}")
    class_f1 = f1_score(all_labels, all_output_indices, labels=[0, 1, 2], average=None)

    with open(f"../output/{output_name}-predictions.txt", "a") as out_file:
        out_file.write(f"{all_output_indices}\n{all_labels}")

    combined_f1 = np.sum(class_f1) / 3
    print(f"\tF1: {class_f1[0], class_f1[1], combined_f1}")
    seen_labels = [all_labels[i] for i in range(len(all_labels)) if all_seen[i] == 1]
    seen_predictions = [all_output_indices[i] for i in range(len(all_labels)) if all_seen[i] == 1]
    unseen_labels = [all_labels[i] for i in range(len(all_labels)) if all_seen[i] == 0]
    unseen_predictions = [all_output_indices[i] for i in range(len(all_labels)) if all_seen[i] == 0]
    class_seen_f1 = f1_score(seen_labels, seen_predictions, labels=[0, 1, 2], average=None)
    combined_f1_seen = np.sum(class_seen_f1) / 3
    print(f"\tF1 for seen: {class_seen_f1[0], class_seen_f1[1], combined_f1_seen}")
    class_unseen_f1 = f1_score(unseen_labels, unseen_predictions, labels=[0, 1, 2], average=None)
    combined_f1_unseen = np.sum(class_unseen_f1) / 3
    print(f"\tF1 for unseen: {class_unseen_f1[0], class_unseen_f1[1], combined_f1_unseen}")
    print(f"\t Seen/unseen for entire test set and annotated test set, respectively: "
          f"{all_seen.count(1)} / {all_seen.count(0)}, {annotated_seen.count(1)} / {annotated_seen.count(0)}")


def integrated_gradients(x, y, x_pad_mask, doc_stopword_mask, allow_backprop=True):
    """Note: GI is equivalent to Integrated Gradients with 1 Riemann step (ig_steps=1)"""
    try:
        x_embeds = model.module.get_inputs_embeds(torch.unsqueeze(x, dim=0))[0]
    except torch.nn.modules.module.ModuleAttributeError:
        x_embeds = model.get_inputs_embeds(torch.unsqueeze(x, dim=0))[0]
    baseline = torch.zeros_like(x_embeds, device=DEVICE)
    for index in cls_sep_indices:
        baseline[index] = x_embeds[index]  # copy [CLS] and [SEP] tokens to baseline
    attributions = torch.zeros(len(x_embeds), device=DEVICE)
    for i in range(ig_steps):
        shifted_input = baseline + (i + 1.) / ig_steps * (x_embeds - baseline)
        shifted_input = torch.unsqueeze(shifted_input, dim=0)
        shifted_loss = model(
            inputs_embeds=shifted_input,
            labels=torch.unsqueeze(y, dim=0),
            pad_mask=torch.unsqueeze(x_pad_mask, dim=0),
            doc_stopword_mask=torch.unsqueeze(doc_stopword_mask, dim=0),
            use_dropout=False,
            token_type_ids=torch.unsqueeze(token_type_ids[0], dim=0),
        )[1]
        derivatives = torch.autograd.grad(
            outputs=shifted_loss,
            inputs=x_embeds,
            grad_outputs=torch.ones_like(shifted_loss).to(x.device),
            create_graph=allow_backprop,
            retain_graph=True
        )[0]
        derivatives = torch.sum(derivatives, dim=1)  # aggregate derivatives at the token level
        attributions = attributions + torch.norm(x_embeds - baseline, dim=-1) * derivatives
    return attributions / ig_steps


def train():
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size, shuffle=True)
    epoch_losses = []
    epoch_combined_f1s = []
    for epoch in range(num_epochs):

        # Prepare attribution visualization file
        if prior != "none":
            html_file = f"../output/{output_name}-{epoch}.html"
            with open(html_file, "w") as out_file:
                out_file.write(visualize.header)

        # Train
        print(f"\nBeginning epoch {epoch}...")
        running_correctness_loss, running_prior_loss = 0., 0.
        for i, data in enumerate(train_loader, 0):
            gc.collect()
            inputs, labels, doc_stopword_mask, attribution_info = data
            has_att_labels, weights, relevance_scores = attribution_info
            pad_mask = get_pad_mask(inputs).to(DEVICE)
            pad_mask.requires_grad = False
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            doc_stopword_mask = doc_stopword_mask.to(DEVICE)
            doc_stopword_mask.requires_grad = False
            outputs, correctness_loss, attentions = model.forward(
                pad_mask=pad_mask,
                doc_stopword_mask=doc_stopword_mask,
                inputs=inputs,
                use_dropout=True,
                token_type_ids=token_type_ids[:len(inputs)],
                labels=labels,
                return_attentions=True
            )
            correctness_loss = torch.mean(correctness_loss)
            attentions = torch.stack(attentions)  # shape=(n_layers, batch_size, n_heads, seq_len, seq_len)
            prior_loss = torch.tensor(0, device=DEVICE)
            for j in range(len(inputs)):
                if prior != "none" and has_att_labels[j]:
                    if prior == "ig":
                        attributions = integrated_gradients(
                            inputs[j],
                            labels[j],
                            x_pad_mask=pad_mask[j],
                            doc_stopword_mask=doc_stopword_mask[j],
                        )
                    elif prior == "att":
                        attributions = torch.sum(
                            attentions[-bert_layers:, j, :, :, :],
                            dim=(0, 1, 2)
                        )  # attention received, by token
                    else:
                        print("Invalid value provided for 'prior'!")
                        exit(1)
                    abs_attributions = torch.abs(attributions)[:2 + train_set.doc_len]
                    attribution_scores = abs_attributions / torch.sum(abs_attributions)
                    weight_tensor, relevance_tensor = weights[j].to(DEVICE), relevance_scores[j].to(DEVICE)
                    prior_loss = prior_loss + lda * \
                                 sum((weight_tensor - attribution_scores) ** 2 * relevance_tensor) \
                                 / sum(relevance_tensor)

                    # Output visualization to file
                    tokens = train_set.tokenizer.convert_ids_to_tokens(inputs[j])
                    att_word_weights = abs_attributions * relevance_tensor
                    gold_word_weights = weight_tensor * relevance_tensor
                    attributions_html = visualize.get_words_html(
                        tokens,
                        att_word_weights.tolist()
                    )
                    weights_html = visualize.get_words_html(
                        tokens,
                        gold_word_weights.tolist()
                    )
                    with open(html_file, "a") as out_file:
                        out_file.write(f"<p>Model attributions:</p>\n{attributions_html}\n")
                        out_file.write(f"<p>Attribution labels:</p>\n{weights_html}\n")
                        out_file.write(f"<p>predicted, actual: {outputs[j].tolist(), labels[j].tolist()}</p>\n")

            prior_loss = prior_loss / sum(has_att_labels)
            running_correctness_loss += correctness_loss.item()
            running_prior_loss += prior_loss.item()
            loss = (prior_loss + correctness_loss) / grad_steps
            loss.backward(retain_graph=True)
            if (i + 1) % grad_steps == 0:
                optimizer.step()
                remove_all_grads()

            # Print running losses every 10 batches
            if (i + 1) % (10 * grad_steps) == 0:
                print(f"Epoch #{epoch + 1} iteration #{i + 1}")
                print(f"\tRunning correctness loss: {running_correctness_loss / (i + 1)}")
                if prior != "none":
                    print(f"\tRunning prior loss: {running_prior_loss / (i + 1)}")

        # Validate
        print("Validating...")
        with torch.no_grad():
            all_labels = []
            all_outputs = []
            all_losses = []
            for i, data in enumerate(dev_loader, 0):
                inputs, labels, doc_stopword_mask, _ = data
                pad_mask = get_pad_mask(inputs).to(DEVICE)
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                doc_stopword_mask = doc_stopword_mask.to(DEVICE)
                all_labels.append(labels)
                outputs, correctness_loss = model.forward(
                    pad_mask=pad_mask,
                    doc_stopword_mask=doc_stopword_mask,
                    inputs=inputs,
                    use_dropout=False,
                    token_type_ids=token_type_ids[:len(inputs)],
                    labels=labels
                )
                correctness_loss = torch.mean(correctness_loss)
                all_losses.append(correctness_loss)
                all_outputs.append(outputs)
            all_labels = torch.cat(all_labels, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)
            correctness_loss = torch.mean(torch.stack(all_losses))
            _, all_preds = torch.max(all_outputs, dim=-1)
            class_f1 = f1_score(all_labels.tolist(), all_preds.tolist(), labels=[0, 1, 2], average=None)
            combined_f1 = np.sum(class_f1) / 3
        print(f"\tLoss: {correctness_loss}")
        print(f"\tF1: {class_f1[0], class_f1[1], combined_f1}")
        epoch_losses.append(correctness_loss.item())
        epoch_combined_f1s.append(combined_f1)

        # Save model with best combined F1 on dev set
        print(epoch_combined_f1s, np.argmax(epoch_combined_f1s))
        if np.argmax(epoch_combined_f1s) == epoch:
            print("Saving model...")
            torch.save(model, f"../output/{output_name}.pt")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)  # helps with debugging
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dev_set = VastReader("../data/VAST/vast_dev.csv", no_stopwords=(model_type == 'bert-joint'))
    cls_sep_indices = [0, dev_set.doc_len + 1, -1]

    # create token_type_ids and token_type_ids_steps (latter is used for Integrated Gradients)
    token_type_ids = [0 for _ in range(dev_set.doc_len + 2)] + [1 for _ in range(dev_set.topic_len + 1)]
    token_type_ids = [token_type_ids for _ in range(batch_size)]
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=DEVICE)
    token_type_ids_steps = torch.stack([token_type_ids[0] for step in range(ig_steps)])
    token_type_ids.requires_grad = False
    token_type_ids_steps.requires_grad = False

    if not load_weights:
        print("Loading data...")
        train_set = VastReader(
            main_csv,
            exclude_from_main="../data/VAST_word_importance/special_datapoints_train.txt",
            word_importance_csv="../data/VAST_word_importance/processed_annotated_train.csv",
            smoothing=None,
            relevance_type=relevance_type,
            use_tf_idf_as_labels=use_tfidf_as_labels,
            no_stopwords=(model_type == 'bert-joint')
        )

        print("Loading model...")
        if model_type == 'bert-joint-new':
            model = BertJoint(doc_len=dev_set.doc_len, fix_bert=False)
        elif model_type == 'bert-joint':
            model = BertJoint(doc_len=dev_set.doc_len, fix_bert=True)
        elif model_type == 'bert-basic':
            model = BertBasic(doc_len=dev_set.doc_len)
        else:
            print("Please specify a valid model type.", file=sys.stderr)
            exit(1)
        if generate_untuned:
            torch.save(model, f"../output/{output_name}.pt")
            exit(0)
        model.to(DEVICE)
        if DEVICE == "cuda":
            model = DataParallel(model)
        optimizer = Adam(model.parameters(), lr=learning_rate)

        train()
    if do_eval:
        model = torch.load(f"../output/{output_name}.pt", map_location=DEVICE)
        print("Evaluating...")
        test_set = VastReader(
            "../data/VAST/vast_test.csv",
            exclude_from_main="../data/VAST_word_importance/special_datapoints_test.txt",
            word_importance_csv="../data/VAST_word_importance/processed_annotated_test.csv",
            smoothing=None,
            relevance_type=relevance_type,
            return_seen=True,
            no_stopwords=(model_type == 'bert-joint')
        )
        evaluate()
