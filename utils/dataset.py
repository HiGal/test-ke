from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer


class ProductDataset(Dataset):
    """
    Датасет заголовков продуктов (понадобится для перевода их в векторную форму)
    """

    def __init__(self, data_path):
        super(ProductDataset, self).__init__()
        self.df = pd.read_csv(data_path, index_col=0)
        self.df = self.df.reset_index(drop=True)  # переопределяем индексы

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row['title']


tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")


def collate_products(batch):
    """
    Функция для формирования батча.
    :param batch: набор заголовков продукта
    :return: BPE токены
    """
    tokens = tokenizer(batch, return_tensors='pt', padding=True).to("cuda:0")

    return tokens
