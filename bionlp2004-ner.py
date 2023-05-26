import datasets
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, pipeline
from bionlp2004_functions import *
import evaluate
import json

def compute_metrics(eval_preds):
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    results = metric.compute(predictions=predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


bionlp2004 = datasets.load_dataset('tner/bionlp2004')

#dictionary to translate numeric labels to labels
labels_to_ids = {
    "O": 0,
    "B-DNA": 1,
    "I-DNA": 2,
    "B-protein": 3,
    "I-protein": 4,
    "B-cell_type": 5,
    "I-cell_type": 6,
    "B-cell_line": 7,
    "I-cell_line": 8,
    "B-RNA": 9,
    "I-RNA": 10
}
#similar to above dictionary in a reversed way
ids_to_labels = {v:k for k,v in labels_to_ids.items()}

#bert base uncased tokenizer is used
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

tokenized_datasets = bionlp2004.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=11)

args = TrainingArguments(
    "bionlp2004_ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = evaluate.load("seqeval")

label_list = list(ids_to_labels.values())

trainer = Trainer(
    model,
    args,
   train_dataset=tokenized_datasets["train"],
   eval_dataset=tokenized_datasets["validation"],
   data_collator=data_collator,
   tokenizer=tokenizer,
   compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("ner_model")

tokenizer.save_pretrained("tokenizer")

config = json.load(open("ner_model/config.json"))

id2label = {str(k):v for k,v in ids_to_labels.items()}
label2id = {v:str(k) for k,v in labels_to_ids.items()}

config["id2label"] = id2label
config["label2id"] = label2id

json.dump(config, open("ner_model/config.json","w"))
model_fine_tuned = AutoModelForTokenClassification.from_pretrained("ner_model")

test_s = " ".join(bionlp2004['test'][2]['tokens'])

nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)
ner_results = nlp(test_s)

print(ner_results)