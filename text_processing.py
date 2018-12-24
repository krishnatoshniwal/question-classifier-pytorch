import re


def load_vocab_dict_from_file(dict_file, pad_at_first=True):
    with open(dict_file, encoding='utf-8') as f:
        words = [w.strip() for w in f.readlines()]
    if pad_at_first and words[0] != '<pad>':
        raise Exception("The first word needs to be <pad> in the word list.")
    vocab_dict = {words[n]: n for n in range(len(words))}
    return vocab_dict


UNK_IDENTIFIER = '<unk>'  # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def sentence2vocab_indices(sentence, vocab_dict):
    if isinstance(sentence, bytes):
        sentence = sentence.decode()
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    # remove .
    if len(words) > 0 and (words[-1] == '.' or words[-1] == '?'):
        words = words[:-1]
    vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
                     for w in words]
    return vocab_indices


PAD_IDENTIFIER = '<pad>'


def preprocess_vocab_indices(vocab_indices, vocab_dict, T):
    # Truncate long sentences
    if len(vocab_indices) > T:
        vocab_indices = vocab_indices[:T]
    # Pad short sentences at the beginning with the special symbol '<pad>'
    if len(vocab_indices) < T:
        vocab_indices = [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices)) + vocab_indices
    return vocab_indices


def preprocess_sentence(sentence, vocab_dict, T):
    vocab_indices = sentence2vocab_indices(sentence, vocab_dict)
    return preprocess_vocab_indices(vocab_indices, vocab_dict, T)


################################################################################
# pos_vocab related functions
################################################################################

def load_pos_vocab_dict_from_file(dict_file):
    with open(dict_file, encoding='utf-8') as f:
        poss = [p.strip() for p in f.readlines()]
    pos_vocab_dict = {poss[n]: n for n in range(len(poss))}
    return pos_vocab_dict


POS_UNK_IDENTIFIER = '<pos_unk>'


def pos_seq2pos_vocab_indices(pos_seq, pos_vocab_dict):
    if isinstance(pos_seq, bytes):
        pos_seq = pos_seq.decode()
    poss = pos_seq.split()  # Just simply split it is fine since the original POS sequence is space joined.
    # remove .
    if len(poss) > 0 and poss[-1] == '.':
        poss = poss[:-1]
    pos_vocab_indices = [(pos_vocab_dict[p] if p in pos_vocab_dict else pos_vocab_dict[POS_UNK_IDENTIFIER])
                         for p in poss]
    return pos_vocab_indices


POS_PAD_IDENTIFIER = '<pos_pad>'


def preprocess_pos_vocab_indices(pos_vocab_indices, pos_vocab_dict, T):
    # Truncate long sentences
    if len(pos_vocab_indices) > T:
        pos_vocab_indices = pos_vocab_indices[:T]
    # Pad short sentences at the beginning with the special symbol '<pos_pad>'
    if len(pos_vocab_indices) < T:
        pos_vocab_indices = [pos_vocab_dict[POS_PAD_IDENTIFIER]] * (T - len(pos_vocab_indices)) + pos_vocab_indices
    return pos_vocab_indices


def preprocess_pos_seq(pos_seq, pos_vocab_dict, T):
    # pos_seq = to_pos_seq(sentence)
    pos_vocab_indices = pos_seq2pos_vocab_indices(pos_seq, pos_vocab_dict)
    return preprocess_pos_vocab_indices(pos_vocab_indices, pos_vocab_dict, T)
