import regex as re
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens.sort(key=len, reverse=True)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        """Construct and return a Tokenizer from serialized files."""
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            vocab = {int(k): bytes(v, "utf-8") for k, v in vocab.items()}

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (bytes([int(token1)]), bytes([int(token2)])) for token1, token2 in merges
            ]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def pretokenize(
        self,
        text: str,
    ) -> list[list[bytes]]:
        # split the text into chunks based on special tokens
        if len(self.special_tokens) > 0:
            pattern = "(" + "|".join(re.escape(st) for st in self.special_tokens) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        # pretokenize the chunks using regex
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                pre_tokens.append([chunk.encode("utf-8")])
                continue

            match = re.finditer(PAT, chunk)
            for m in match:
                token_bytes = [bytes([b]) for b in m.group().encode("utf-8")]
                pre_tokens.append(token_bytes)

        return pre_tokens

    def encode(self, text: str) -> list[int]:
        """Encode input text into token IDs."""
        pre_tokens = self.pretokenize(text)
        for merge in self.merges:
            for pre_token in pre_tokens:
                i = 0
                while i < len(pre_token) - 1:
                    if (pre_token[i], pre_token[i + 1]) == merge:
                        pre_token[i] = pre_token[i] + pre_token[i + 1]
                        del pre_token[i + 1]
                    else:
                        i += 1

        token_ids = []
        for pre_token in pre_tokens:
            for token_bytes in pre_token:
                for token_id, vocab_bytes in self.vocab.items():
                    if vocab_bytes == token_bytes:
                        token_ids.append(token_id)
                        break
        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        """Lazily yield token IDs from an iterable of strings."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text."""
        token_bytes = bytes()
        for token_id in ids:
            if token_id in self.vocab:
                token_bytes += self.vocab[token_id]
            else:
                raise ValueError(f"Token ID {token_id} not found in vocabulary.")

        text = token_bytes.decode("utf-8", errors="replace")
        return text
