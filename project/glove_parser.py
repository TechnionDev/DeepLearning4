from enum import Enum
import pickle
import torch
import csv


class GloveDimSize(Enum):
    FIFTY = 0
    HUNDRED = 1
    TWO_HUNDRED = 2
    THREE_HUNDRED = 3


dim_to_mapping_size = {
    GloveDimSize.FIFTY: 50,
    GloveDimSize.HUNDRED: 100,
    GloveDimSize.TWO_HUNDRED: 200,
    GloveDimSize.THREE_HUNDRED: 300
}


def read_glove_dim(embedding_size: GloveDimSize):
    words = []
    vectors = []
    word_to_idx = {}
    idx = 0
    with open(f"embedding/glove.6B.{dim_to_mapping_size[embedding_size]}d.txt") as f:
        for line in f:
            line = line.split(" ")
            word = line[0]
            vec = [float(num) for num in line[1:]]
            word_to_idx[word] = idx
            words.append(word)
            vectors.append(vec)
            idx = idx + 1
        vectors = torch.FloatTensor(vectors)
        torch.save(vectors, f"embedding_parsed/glove.6B.{dim_to_mapping_size[embedding_size]}d.pt")
        # with open('eggs.csv', 'w', newline='') as csvfile:

        with open(f'embedding_parsed/glove.6B.{dim_to_mapping_size[embedding_size]}d_word_to_idx.pkl', 'wb') as f:
            pickle.dump(word_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)


