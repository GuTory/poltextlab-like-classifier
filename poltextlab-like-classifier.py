import os
import pandas as pd
import numpy as np
import gzip
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from article_dataset import ArticleDataset
from sklearn.utils import shuffle

from transformers import BertForSequenceClassification

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import BertTokenizerFast as BertTokenizer

    LABEL = 'major_topic_pred_index'
    ARTICLE = 'article'

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
        
    def reduce(_number=6):
        # csak a test
        test = os.path.join(os.getcwd(), 'test')
        df = read_folder(test)
        df.drop('uuid', axis=1, inplace=True)
        return shuffle(df.groupby('major_topic_pred').apply(lambda x: x.sample(n=min(_number, len(x)))).reset_index(drop=True))


    df_train = reduce()
    df_test = reduce()
    df_eval = reduce()

    print(df_train.index)
    print(df_test.index)
    print(df_eval.index)

    labels = df_train['major_topic_pred'].unique().tolist()
    number_of_labels = len(labels)
    id_to_label = {_id: label for _id, label in enumerate(labels)}
    label_to_id = {label: _id for _id, label in enumerate(labels)}


    df_train["major_topic_pred_index"] = df_train['major_topic_pred'].map(lambda x: label_to_id[x])
    df_test["major_topic_pred_index"] = df_test['major_topic_pred'].map(lambda x: label_to_id[x])
    df_eval["major_topic_pred_index"] = df_eval['major_topic_pred'].map(lambda x: label_to_id[x])

    print(df_eval.info())

    print(df_train.head())

    BERT_MODEL_NAME = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, return_tensors='pt')

    MAX_TOKEN_COUNT = 512
    MAX_LEN = 512
    BATCH_SIZE = 8
    EPOCHS = 60

    def create_dataloader(df, tokenizer, max_token_len, batch_size):
        dataset = ArticleDataset(
            df,
            tokenizer=tokenizer,
            max_token_len=max_token_len
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1
        )


    data_loader_train = create_dataloader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    data_loader_test = create_dataloader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    data_loader_eval = create_dataloader(df_eval, tokenizer, MAX_LEN, BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, return_dict=True, num_labels=number_of_labels)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    accuracy_per_epoch = []
    early_stopping_epochs = 5

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_labels = 0
        prev_loss = np.inf
        for batch in data_loader_train:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            correct_labels += (logits.argmax(dim=1) == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if total_loss < prev_loss - 0.001:
            prev_loss = total_loss
            counter = 0
        else:
            counter += 1
            if counter >= early_stopping_epochs:
                print(f'Early stopping at epoch {epoch}')
                break
            

        accuacy = correct_labels / len(df_train)
        accuracy_per_epoch.append(accuacy)
        print(f"Epoch: {epoch}, Loss: {loss.item()} Accuracy: {accuacy}")
        
    print(accuracy_per_epoch)
    plt.plot(accuracy_per_epoch)

    model.eval()
    eval_loss = 0
    correct_labels = 0

    with torch.no_grad():
        for batch in data_loader_train:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.item()
            #print(f"{logits})")
            print(f"{logits.argmax(dim=1)} : {labels}")
            correct_labels += (logits.argmax(dim=1) == labels).sum().item()

    # Calculate evaluation metrics
    eval_len = len(data_loader_train.dataset)
    eval_loss /= eval_len
    accuracy = correct_labels / eval_len

    print(f"Evaluation Loss: {eval_loss}")
    print(f"Accuracy: {accuracy}")

    with torch.no_grad():
        for batch in data_loader_test:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.item()
            #print(f"{logits})")
            print(f"{logits.argmax(dim=1)} : {labels}")
            correct_labels += (logits.argmax(dim=1) == labels).sum().item()
            
    # Calculate evaluation metrics
    eval_len = len(data_loader_test.dataset)
    eval_loss /= eval_len
    accuracy = correct_labels / eval_len

    print(f"Evaluation Loss: {eval_loss}")
    print(f"Accuracy: {accuracy}")

    with torch.no_grad():
        for batch in data_loader_eval:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.item()
            #print(f"{logits})")
            print(f"{logits.argmax(dim=1)} : {labels}")
            correct_labels += (logits.argmax(dim=1) == labels).sum().item()
            
    # Calculate evaluation metrics
    eval_len = len(data_loader_eval.dataset)
    eval_loss /= eval_len
    accuracy = correct_labels / eval_len

    print(f"Evaluation Loss: {eval_loss}")
    print(f"Accuracy: {accuracy}")

    print("Saving model")
    model_path = 'poltextlab-like-classification-model'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Load saved model and tokenizer
    loaded_model = BertForSequenceClassification.from_pretrained(model_path)
    loaded_tokenizer = BertTokenizer.from_pretrained(model_path)

    # Perform inference with the loaded model
    print("Performing inference with the loaded model")
    inputs = tokenizer("A gyár nettó profitja 439 millió euróra csökkent az elõzõ év harmadik negyedévében elért 903 millió euróhoz képest. Értékesítési árbevétele ezzel szemben 21,2 milliárd euróra emelkedett, ami 2,2 százalékos növekedést jelent."
    "A német autógyár, amely olyan autómárkákat birtokol, mint az Audi, Seat, Skoda és a VW, nyereségének csökkenését az európai, illetve észak- és dél amerikai piacokon tapasztalható, idõsödõ modelljei iránti keresletcsökkenéssel magyarázta."
    "(Üzleti Negyed)"
    "Ajánlat:"
    "Volkswagen"
    "Korábban:", return_tensors="pt")
    outputs = loaded_model(**inputs)
    predictions = outputs.logits
    id = torch.argmax(predictions, dim=1).item()
    print(id_to_label[id])


if __name__ == '__main__':
    run()