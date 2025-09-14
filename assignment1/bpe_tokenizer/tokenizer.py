import json
import regex as re
from typing import List, Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.special_tokens = special_tokens or []

        self.encoder = {token_bytes: token_id for token_id,
                        token_bytes in self.vocab.items()}
        self.merge_rules = {}
        for p1, p2 in merges:
            id1 = self.encoder.get(p1)
            id2 = self.encoder.get(p2)
            if id1 is None or id2 is None:
                continue

            new_token_bytes = p1 + p2
            new_id = self.encoder.get(new_token_bytes)
            if new_id is not None:
                self.merge_rules[(id1, id2)] = new_id

        if self.special_tokens:
            sorted_special_tokens = sorted(
                self.special_tokens, key=len, reverse=True)

            special_pattern = "|".join(re.escape(tok)
                                       for tok in sorted_special_tokens)
            self.special_splitter = re.compile(f"({special_pattern})")
            self.special_tokens_set = set(self.special_tokens)

        self.word_splitter = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None) -> 'Tokenizer':
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_str_keys = json.load(f)
            vocab = {int(k): v.encode("utf-8")
                     for k, v in vocab_str_keys.items()}

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_str = [tuple(line.strip().split(" ", 1)) for line in f]
            merges = [(p1.encode("utf-8"), p2.encode("utf-8"))
                      for p1, p2 in merges_str]

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, ids: List[int]) -> List[int]:
        while len(ids) >= 2:
            best_pair_position = -1
            min_merge_rank = float('inf')

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                if pair in self.merge_rules:
                    rank = self.merge_rules[pair]
                    if rank < min_merge_rank:
                        min_merge_rank = rank
                        best_pair_position = i

            if best_pair_position == -1:
                break

            pair_to_merge = (ids[best_pair_position],
                             ids[best_pair_position + 1])
            new_id = self.merge_rules[pair_to_merge]
            ids = ids[:best_pair_position] + \
                [new_id] + ids[best_pair_position + 2:]

        return ids

    def encode(self, text: str) -> List[int]:
        final_ids = []

        chunks_to_process = [text]
        if self.special_tokens:
            chunks_to_process = self.special_splitter.split(text)

        for chunk in filter(None, chunks_to_process):
            if self.special_tokens and chunk in self.special_tokens_set:
                special_token_bytes = chunk.encode("utf-8")
                final_ids.append(self.encoder[special_token_bytes])
            else:
                for word_match in self.word_splitter.finditer(chunk):
                    word_bytes = word_match.group(0).encode("utf-8")
                    initial_ids = [
                        self.encoder[bytes([b])] for b in word_bytes]
                    merged_ids = self._apply_merges(initial_ids)
                    final_ids.extend(merged_ids)

        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_string in iterable:
            yield from self.encode(text_string)

    def decode(self, ids: list[int]) -> str:
        all_bytes = b"".join(self.vocab.get(i, b'') for i in ids)
        return all_bytes.decode("utf-8", errors="replace")
