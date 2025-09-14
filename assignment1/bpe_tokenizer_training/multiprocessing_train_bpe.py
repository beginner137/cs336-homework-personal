import regex as re
from collections import defaultdict
from typing import List, BinaryIO
import os
import multiprocessing
import functools


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 1024 * 1024 

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  
        while True:
            mini_chunk = file.read(mini_chunk_size)  

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


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


def process_chunk(input_path, start, end):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end-start)
        return chunk_bytes.decode("utf-8", errors="ignore")


def parallel_pre_tokenize_and_count(text_chunk: str, special_tokens: list[str]) -> defaultdict[tuple, int]:
    local_word_freq = defaultdict(int)

    sub_chunks = [text_chunk]
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        sub_chunks = re.split(pattern, text_chunk)

    for chunk in filter(None, sub_chunks):
        words_by_chunk = pre_tokenize_into_words(chunk)
        for word in words_by_chunk:
            local_word_freq[tuple(word)] += 1

    return local_word_freq


def train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, "rb") as f:
        num_processes = multiprocessing.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_boundaries = list(zip(boundaries[:-1], boundaries[1:]))
    read_process_func = functools.partial(process_chunk, input_path)

    with multiprocessing.Pool(processes=num_processes) as pool:
        text_chunks = pool.starmap(read_process_func, chunk_boundaries)

        vocab = {i: bytes([i]) for i in range(256)}
        if special_tokens:
            special_tokens_sorted = sorted(
                special_tokens, key=len, reverse=True)
            for i, token_str in enumerate(special_tokens_sorted):
                tok_id = 256 + i
                vocab[tok_id] = token_str.encode("utf-8")
        else:
            special_tokens_sorted = []

        tokenize_process_func = functools.partial(
            parallel_pre_tokenize_and_count, special_tokens=special_tokens_sorted)

        list_of_freq_dicts = pool.map(tokenize_process_func, text_chunks)

        word_freqs = defaultdict(int)
        for local_dict in list_of_freq_dicts:
            for word, freq in local_dict.items():
                word_freqs[word] += freq

    merge_order = []
    next_id = len(vocab)
    num_merges = vocab_size - len(vocab)

    counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        prev_pairs = list(zip(word, word[1:]))        
        for p in prev_pairs:
            counts[p] += freq
        for unique_p in set(prev_pairs):
            pair_to_words[unique_p].add(word)

    for _ in range(num_merges):
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

        affected_words = list(pair_to_words[pair])

        for word in affected_words:
            freq = word_freqs[word]
            word_list = list(word)

            if len(word) >= 2:
                for i1, i2 in zip(word, word[1:]):
                    counts[(i1, i2)] -= freq
                    if counts[(i1, i2)] == 0:
                        del counts[(i1, i2)]

            new_list = merge(word_list, pair, new_index)
            new_word = tuple(new_list)

            if len(new_word) >= 2:
                for i1, i2 in zip(new_word, new_word[1:]):
                    counts[(i1, i2)] += freq

            del word_freqs[word]
            word_freqs[new_word] += freq

            old_pairs = list(zip(word, word[1:]))
            for op in set(old_pairs):
                pair_to_words[op].discard(word)
                if not pair_to_words[op]:
                    del pair_to_words[op]

            new_pairs = list(zip(new_word, new_word[1:]))
            for np in set(new_pairs):
                pair_to_words[np].add(new_word)

    return vocab, merge_order