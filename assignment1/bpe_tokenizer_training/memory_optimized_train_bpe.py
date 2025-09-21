import regex as re
from collections import defaultdict
from typing import List, BinaryIO
import os
import multiprocessing
import functools
from tqdm import tqdm  # Import tqdm for the progress bar


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunks the file into parts that can be counted independently.
    This function remains largely the same but is a key part of the parallel processing setup.
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 1024 * 1024  # Read 1MB at a time

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
    """Merges a pair of tokens in a list of token indices in-place, but returns the modified list."""
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
    """Splits a string into a list of words, where each word is a list of UTF-8 bytes."""
    PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    words_as_byte_lists = [list(m.group(0).encode("utf-8"))
                           for m in PAT.finditer(string)]
    return words_as_byte_lists


def process_chunk(input_path: str, start: int, end: int) -> str:
    """Reads a chunk of a file from disk."""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        return chunk_bytes.decode("utf-8", errors="ignore")


def parallel_pre_tokenize_and_count(text_chunk: str, special_tokens: list[str]) -> defaultdict[tuple, int]:
    """
    Tokenizes a text chunk and counts the frequency of each 'word'.
    This version correctly splits by special tokens, excluding them from the merge analysis.
    """
    local_word_freq = defaultdict(int)

    # Split the text by the special tokens. This removes the special tokens from the
    # chunks of text that will be processed for pair counting, which is the correct behavior.
    sub_chunks = [text_chunk]
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        sub_chunks = re.split(pattern, text_chunk)

    # Tokenize and count words for each resulting sub-chunk
    for chunk in filter(None, sub_chunks):
        words_by_chunk = pre_tokenize_into_words(chunk)
        for word in words_by_chunk:
            local_word_freq[tuple(word)] += 1

    return local_word_freq


def get_pair_counts(word_freqs: defaultdict[tuple, int]) -> defaultdict[tuple, int]:
    """Calculates the frequency of adjacent pairs of tokens."""
    pair_counts = defaultdict(int)
    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
    return pair_counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    max_memory_gb: int = 4
):
    """
    Trains a BPE tokenizer from a text file with memory optimizations, progress tracking,
    batched processing, and efficient merge updates to handle large files and speed up training.
    """
    # Setup for parallel processing
    num_processes = multiprocessing.cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_boundaries = list(zip(boundaries[:-1], boundaries[1:]))

    # Batching Logic to Control Memory Usage
    max_memory_bytes = max_memory_gb * 1024**3
    batches = []
    current_batch_boundaries = []
    current_batch_size = 0

    print("Calculating batches to fit within memory limits...")
    for start, end in chunk_boundaries:
        chunk_size = end - start
        if current_batch_boundaries and current_batch_size + chunk_size > max_memory_bytes:
            batches.append(current_batch_boundaries)
            current_batch_boundaries = []
            current_batch_size = 0
        current_batch_boundaries.append((start, end))
        current_batch_size += chunk_size
    if current_batch_boundaries:
        batches.append(current_batch_boundaries)

    # Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    if special_tokens:
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        for i, token_str in enumerate(special_tokens_sorted):
            tok_id = 256 + i
            vocab[tok_id] = token_str.encode("utf-8")
    else:
        special_tokens_sorted = []

    # Process each batch sequentially
    word_freqs = defaultdict(int)
    print(
        f"Processing {len(chunk_boundaries)} total chunks in {len(batches)} batch(es).")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Outer progress bar for batches
        for i, batch in enumerate(tqdm(batches, desc="Processing Batches")):
            # Step 1: Read the chunks for the current batch into memory
            read_process_func = functools.partial(process_chunk, input_path)
            text_chunks_batch = pool.starmap(read_process_func, batch)

            # Step 2 & 3: Perform parallel tokenization and aggregate results with a nested progress bar
            tokenize_process_func = functools.partial(
                parallel_pre_tokenize_and_count, special_tokens=special_tokens_sorted
            )

            # --- New nested progress bar for chunks within the batch ---
            pbar_inner = tqdm(
                pool.imap_unordered(tokenize_process_func, text_chunks_batch),
                total=len(batch),
                desc=f"Tokenizing Batch {i + 1}",
                leave=False  # Hides the inner bar once completed
            )
            for local_dict in pbar_inner:
                for word, freq in local_dict.items():
                    word_freqs[word] += freq

            # Clean up to explicitly release memory
            del text_chunks_batch

    # Prepare mutable words list and indices for efficient updates
    words = [(list(word), freq)
             for word, freq in word_freqs.items() if freq > 0]
    del word_freqs  # Free memory

    # Calculate initial pair counts and pair-to-words index
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for wi, (word_list, freq) in enumerate(words):
        if len(word_list) < 2:
            continue
        for pair in zip(word_list, word_list[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(wi)

    merge_order = []
    next_id = len(vocab)
    num_merges = vocab_size - len(vocab)

    # --- Main training loop with efficient updates ---
    print("Starting BPE merges...")
    for _ in tqdm(range(num_merges), desc="BPE Merges"):
        if not pair_counts:
            print("No more pairs to merge. Stopping early.")
            break

        best_pair = max(pair_counts, key=lambda p: (
            pair_counts[p], (vocab.get(p[0]), vocab.get(p[1]))))

        new_index = next_id
        next_id += 1
        merge_order.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_index] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # Get affected word indices (efficient: only those with the pair)
        # Copy to avoid modification during iteration
        affected_wis = list(pair_to_words[best_pair])

        for wi in affected_wis:
            word_list, freq = words[wi]

            # Compute old pairs
            old_pairs = list(zip(word_list, word_list[1:]))

            # Remove from old pair indices
            for pair in old_pairs:
                pair_to_words[pair].discard(wi)

            # Update pair counts for old pairs
            for pair in old_pairs:
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            # Merge in-place (but use the function, which returns new list; assign back)
            word_list[:] = merge(word_list, best_pair, new_index)

            # Compute new pairs
            new_pairs = list(zip(word_list, word_list[1:]))

            # Add to new pair indices
            for pair in new_pairs:
                pair_to_words[pair].add(wi)

            # Update pair counts for new pairs
            for pair in new_pairs:
                pair_counts[pair] += freq

    return vocab, merge_order
