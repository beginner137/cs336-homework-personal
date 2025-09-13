import regex as re
from collections import defaultdict
from typing import List


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


def pre_tokenize_into_words(string: str) -> List[List[int]]:
    PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    words_as_byte_lists = []
    for m in PAT.finditer(string):
        words_as_byte_lists.append(list(m.group(0).encode("utf-8")))
    return words_as_byte_lists


def train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    vocab = {i: bytes([i]) for i in range(256)}
    if special_tokens:
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        for i, token_str in enumerate(special_tokens):
            tok_id = 256 + i
            vocab[tok_id] = token_str.encode("utf-8")

    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(pattern, text)
    else:
        chunks = [text]
    chunks = [c for c in chunks if c]

    words_by_chunk = [pre_tokenize_into_words(chunk) for chunk in chunks]

    merge_order = []
    next_id = len(vocab)
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        counts = defaultdict(int)
        for chunk_of_words in words_by_chunk:
            for word in chunk_of_words:
                if len(word) >= 2:
                    for i1, i2 in zip(word, word[1:]):
                        counts[(i1, i2)] += 1
        if not counts:
            break

        pair = max(
            counts.items(),
            key=lambda kv: (kv[1], (vocab[kv[0][0]], vocab[kv[0][1]]))
        )[0]

        a, b = pair
        new_index = next_id
        next_id += 1

        merge_order.append((vocab[a], vocab[b]))
        vocab[new_index] = vocab[a] + vocab[b]

        new_words_by_chunk = []
        for chunk_of_words in words_by_chunk:
            new_chunk = [merge(word, pair, new_index)
                         for word in chunk_of_words]
            new_words_by_chunk.append(new_chunk)
        words_by_chunk = new_words_by_chunk

    return vocab, merge_order