import gzip
import json
import os
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer, pipeline
from torch import cuda
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


device = 'cuda' if cuda.is_available() else 'cpu'


class DataLoader(Dataset):
    def __init__(self, encodings, _labels):
        self.encodings = encodings
        self.labels = _labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def read_folder(folder_path):
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl.gz'):
            with gzip.open(os.path.join(folder_path, filename), 'rt', encoding='utf-8') as file:
                for line in file:
                    json_data = json.loads(line)
                    df = pd.DataFrame(json_data)
                    dataframes.append(df)
    if dataframes:
        aggregated_df = pd.concat(dataframes, ignore_index=True)
        return aggregated_df
    else:
        print("No jsonl files found in the directory.")
        return None
    

def reduce():
    # csak a test
    test = os.path.join(os.getcwd(), 'test')
    df = read_folder(test)
    df.drop('uuid', axis=1, inplace=True)
    return df.groupby('major_topic_pred').apply(lambda x: x.sample(n=min(10, len(x)))).reset_index(drop=True)


def compute_metrics(predictions):
    labels = predictions.label_ids
    preds = predictions.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


def predict(model, text, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    inputs = tokenizer(text, padding='longest', truncation=True, max_length=512, return_tensors='pt').to(device)

    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    pred_label_idx = probs.argmax()

    pred_label = model.config.id2label[pred_label_idx.item()]
    return probs.cpu(), pred_label_idx.cpu(), pred_label


def main():
    reduced_df = reduce()
    df = shuffle(reduced_df)
    labels = df['major_topic_pred'].unique().tolist()
    number_of_labels = len(labels)
    id_to_label = {_id: label for _id, label in enumerate(labels)}
    label_to_id = {label: _id for _id, label in enumerate(labels)}
    print(id_to_label)
    df['major_topic_pred_index'] = df['major_topic_pred'].map(lambda x: label_to_id[x])

    model = BertForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased",
        num_labels=number_of_labels,
        id2label=id_to_label,
        label2id=label_to_id
    )
    model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    size = df.shape[0]
    half = size // 2
    three_fourth = (3 * size) // 4
    train_texts = list(df['article'][:half])
    val_texts = list(df['article'][half:three_fourth])
    test_texts = list(df['article'][three_fourth:])
    print(len(train_texts), len(val_texts), len(test_texts))
    train_labels = list(df['major_topic_pred_index'][:half])
    val_labels = list(df['major_topic_pred_index'][half:three_fourth])
    test_labels = list(df['major_topic_pred_index'][three_fourth:])
    print(len(train_labels), len(val_labels), len(test_labels))
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataloader = DataLoader(train_encodings, train_labels)
    val_dataloader = DataLoader(val_encodings, val_labels)
    test_dataset = DataLoader(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir='./z',
        do_train=True,
        do_eval=True,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_strategy='steps',
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy='steps',
        eval_steps=50,
        save_strategy='steps',
        load_best_model_at_end=True,
        gradient_accumulation_steps=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader,
        compute_metrics=compute_metrics
    )
    trainer.train()

    model_path = 'poltextlab-like-classification-model'
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer=BertTokenizerFast.from_pretrained(model_path)
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    print(nlp("Egyre t\u00f6bb a t\u00f6meges H1N1-megbetegedes, sok a beteg a korhazakban, betegseg, rak, illness angolul"))



if __name__ == "__main__":
    main()