import os
import torch
from torch.utils.data import Dataset


class Dictionary(object):

    def __init__(self, path = None):
        self.word2idx = {"<unk>": 0}
        self.word_num = {}
        self.idx2word = ["<unk>"]
        if path is not None:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line.strip()) == 0:
                        continue
                    words = ['<ini>'] + list(line.strip()) + ['<end>']
                    for word in words:
                        self._add_word(word)

    def _add_word(self, word):
        if word not in self.word_num:
            self.word_num[word] = 1
        else:
            self.word_num[word] += 1
        if word not in self.word2idx and self.word_num[word] > 1:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def find_idx(self, word):
        if word not in self.word2idx:
            word = "<unk>"
        return self.word2idx[word]

    def find_word(self, idx):
        return self.idx2word[idx]

    def translate(self, indexes):
        words = [self.find_word(idx) if idx!=self.word2idx["<ini>"] and idx!=self.word2idx["<end>"] else ""
                 for idx in indexes]
        return ''.join(words)

    def __len__(self):
        return len(self.idx2word)


class QinghuaDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        self.dictionary = Dictionary(path)
        self.data = self._load_data(path)

    def _load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                if len(line.strip()) == 0:
                    continue
                words = ['<ini>'] + list(line.strip()) + ['<end>']
                data.append(torch.LongTensor([self.dictionary.find_idx(word) for word in words]))
        return data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = QinghuaDataset("./data.txt")
    print(dataset.dictionary.translate(dataset[233]))
