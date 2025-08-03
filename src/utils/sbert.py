import torch
import torch.nn.functional as F

from src.utils.helpers import mean_pooling


def extract_sentence_embeddings(dataset, model, tokenizer, batch_size=32, device='cpu'):
    """
    Extracts embeddings from the model for the given data.
    """
    all_embeddings = []
    for examples in dataset.iter(batch_size=batch_size):
        inputs = tokenizer(examples['question'], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        all_embeddings.append(sentence_embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings
