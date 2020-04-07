from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from itertools import islice
import torch
import torch.nn.functional as F

def get_query_embedding(tokenizer, model, query):
    # sent model works differently
    if tokenizer is None:
        return torch.tensor(model.encode([query]))
    
    query_enc = tokenizer.encode(query, add_special_tokens=True)
    query_enc = torch.tensor([query_enc])
    with torch.no_grad():
        query_output = model(query_enc)
    return query_output[0][:, 0, :]

def find_top_n_similar(embeddings, query_embedding, titles, n=20):
    similarity = F.cosine_similarity(embeddings, query_embedding)
    index_sorted = torch.argsort(similarity, descending=True)
    return index_sorted, [titles[i] for i in index_sorted.tolist()[:n]]

def get_tsne_embeddings(embedding):
    embeddings_np = embedding.numpy()
    centered = embeddings_np - embeddings_np.mean(axis=0)
    pca = PCA(n_components=50)
    components = pca.fit_transform(centered)
    tsne = TSNE()
    result = tsne.fit_transform(components)
    return result

def get_encodings_drop_long(text, tokenizer, max_length=30):
    encoded = [tokenizer.encode(title, add_special_tokens=True) for title in text]
    padded = []
    # get rid of titles longer than 30 tokens
    dropped = 0
    indices_to_drop = []
    for index, s in enumerate(encoded):
        if len(s) > max_length:
            indices_to_drop.append(index)
            dropped += 1
            continue
        padded.append(s)
        for i in range(len(s), max_length):
            padded[-1].append(0)
    print("Dropped {} titles".format(dropped))
    return torch.tensor(padded), indices_to_drop

def drop_from_lists(lists, indices):
    for index in sorted(indices, reverse=True):
        for l in lists:
            l.pop(index)

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def cosine_similarity(first, second):
    first = first.squeeze()
    second = second.squeeze()
    with torch.no_grad():
        numerator = torch.dot(first, second)
        denominator = torch.norm(first) * torch.norm(second)
        return (numerator / denominator).item()

def sentence_embedding(tokenizer, model, sentence, average=False):
    # sent model works differently
    if tokenizer is None:
        return torch.tensor(model.encode([sentence]))
        
    encoded = tokenizer.encode(sentence, add_special_tokens=True)
    with torch.no_grad():
         output = model(torch.tensor([encoded]))
    if average:
        return output[0].squeeze().mean(axis=0)
    return output[0][:, 0, :].squeeze()