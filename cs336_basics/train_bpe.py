import re

def pretokenize(
    input_path: str,
    special_tokens: list[str]
) -> list[bytes]:
    pattern = re.split("|".join(re.escape(st) for st in special_tokens))
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
        pre_tokens = re.split(pattern, text)

    return pre_tokens


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = [bytes([i]) for i in range(256)]
    vocab += [st.encode("utf-8") for st in special_tokens]

    merges = []

    pre_tokens = pretokenize(input_path, special_tokens)

    while len(vocab) < vocab_size:
        pairs = {}
        for pt, cnt in pre_tokens.items():
            pt_bytes = list(pt)
            for i in range(len(pt_bytes) - 1):
                p = (pt_bytes[i], pt_bytes[i+1])
                pairs[p] = pairs.get(p, 0) + cnt

        max_pair, _ = max(pairs.items(), key=lambda item: (item[1], item[0]))
        vocab.append(max_pair[0] + max_pair[1])
        merges.append((max_pair[0], max_pair[1]))

        for pt in list(pre_tokens.keys()):
            pt_bytes = list(pt)
            length = len(pt_bytes)
            new_pt_bytes = []

            i = 0
            while i < length:
                if i < length - 1 and (pt_bytes[i], pt_bytes[i+1]) == max_pair:
                    new_pt_bytes.append(pt_bytes[i]+pt_bytes[i+1])
                    i += 1
                else:
                    new_pt_bytes.append(pt_bytes[i])
                i += 1

            new_pt = tuple(new_pt_bytes)
            pre_tokens[new_pt] = pre_tokens.pop(pt)

    return (vocab, merges)
