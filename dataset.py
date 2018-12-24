import torch.utils.data as data
from utils import load_obj
import text_processing as text_p
from functools import partial
import numpy as np

#
# def collate_fn(device, batch):
#     sentences = [bat[0] for bat in batch]
#     target = [bat[1] for bat in batch]
#     sentences = torch.Tensor(sentences, device=device)
#     target = torch.Tensor(target, device=device)
#     return sentences, target


def get_dataloaders(cfg):
    word_count = cfg.word_count
    vocab_file_fn = cfg.vocab_file_fn
    train_qdb_fn = cfg.train_qdb_fn
    val_qdb_fn = cfg.val_qdb_fn
    test_qdb_fn = cfg.test_qdb_fn

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    embed_matrix = text_p.load_vocab_dict_from_file(vocab_file_fn)

    train_dataset = Dataset(train_qdb_fn, embed_matrix, word_count)
    val_dataset = Dataset(val_qdb_fn, embed_matrix, word_count)
    test_dataset = TestDataset(test_qdb_fn, embed_matrix, word_count)

    # collate_fn_part = partial(collate_fn, cfg.device)

    train_dataloader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,)
                                       # collate_fn=collate_fn_part)
    val_dataloader = data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,)
    test_dataloader = data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,)

    return train_dataloader, val_dataloader, test_dataloader


class Dataset(data.Dataset):
    def __init__(self, qdb_fn, vocab_file, word_count):
        self.vocab_file = vocab_file
        self.qdb = load_obj(qdb_fn)
        self.word_count = word_count

    def __getitem__(self, index):
        sent = self.qdb[index][0]
        sentence = text_p.preprocess_sentence(sent, self.vocab_file, self.word_count)
        target = self.qdb[index][1]

        sentence = np.array(sentence).astype(np.int64)
        target = np.array(target).astype(np.float32)

        return sentence, target

    def __len__(self):
        return len(self.qdb)


class TestDataset(data.Dataset):
    def __init__(self, qdb_fn, vocab_file, word_count):
        self.vocab_file = vocab_file
        self.qdb = load_obj(qdb_fn)
        self.word_count = word_count

    def __getitem__(self, index):
        sent = self.qdb[index]
        sentence = text_p.preprocess_sentence(sent, self.vocab_file, self.word_count)
        sentence = np.array(sentence).astype(np.int64)
        return sentence

    def __len__(self):
        return len(self.qdb)

