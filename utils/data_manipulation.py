from torch.utils.data import DataLoader
from utils.dataset import ProductDataset, collate_products
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import numpy as np


def create_products_vectors(data_path, out_path, batch_size=32):
    dataset = ProductDataset(f"{data_path}")
    data_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_products)
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
    model = model.cuda()

    vectors = []
    model.eval()
    for batch_tokens in tqdm(data_dataloader):
        with torch.no_grad():
            out = model(**batch_tokens)
            # вектор предложения это среднее от всех векторов токена
            # (https://huggingface.co/DeepPavlov/rubert-base-cased-sentence)
            out = torch.mean(out.last_hidden_state, dim=1)
            vectors.append(out.cpu())
    product_vectors = torch.cat(vectors)
    np.save(f"{out_path}", product_vectors.numpy())
    print("Done!")


def load_product_vectors(vectors_path):
    product_vectors = np.load('product_vectors.npy')
    return product_vectors
