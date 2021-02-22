from typing import List

import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import os

data = pd.read_csv("data.csv", index_col=0)
data = data.reset_index(drop=True)

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
model = model.cuda()
model.eval()


def create_index():
    """
    Функция создания faiss индекса (если не создан) для дальнейшего поиска
    :return:
    """
    n_clusters = 32
    if os.path.exists("faiss_index"):
        print("Index is already created. Loading index")
        index = faiss.read_index("faiss_index")
        return index
    product_vectors = np.load("product_vectors.npy").astype('float32')
    dimension = product_vectors.shape[1]
    quantiser = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantiser, dimension, n_clusters, faiss.METRIC_L2)
    index.train(product_vectors)
    index.add(product_vectors)
    faiss.write_index(index, "faiss_index")  # сохраняем полученный индекс в файл
    return index


def get_5_nearest(query: str) -> List[str]:
    """
    По заданному поиску выводим список товаров, ближайших к поисковому запросу
    :param query: текстовый поисковой запрос
    :return: список товаров, удовлетворяющих поисковому запроосу
    """
    query_tokens = tokenizer(query, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        out = model(**query_tokens)
        query_vector = torch.mean(out.last_hidden_state, dim=1).cpu().numpy()
    distances, indices = index.search(query_vector, k=5)
    return data.iloc[indices[0]].title.tolist()


if __name__ == '__main__':
    index = create_index()
    query = ""
    while True:
        query = input("Введите поисковой запрос: ")
        if query == "exit":
            break
        print(get_5_nearest(query))
