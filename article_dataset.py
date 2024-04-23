import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast as BertTokenizer

LABEL = 'major_topic_pred_index'
ARTICLE = 'article'

# Dataset a Cikkekhez
class ArticleDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        max_token_len: int = 128
    ) -> None:
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        article = data_row[ARTICLE]
        label = data_row[LABEL]

        encoding = self.tokenizer.encode_plus(
            article,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            article=article,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            label=torch.tensor(label, dtype=torch.long)
        )