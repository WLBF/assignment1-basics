import regex as re

def pretokenize(
    text: str,
    special_tokens: list[str]
) -> list[list[bytes]]:
    # remove special tokens from the text and split the text into chunks
    if len(special_tokens) > 0:
        pattern = "|".join(re.escape(st) for st in special_tokens)
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    # pretokenize the chunks using regex
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = []
    for chunk in chunks:
        match = re.finditer(PAT, chunk)
        for m in match:
            pre_tokens.append([bytes([b]) for b in m.group().encode("utf-8")])

    return pre_tokens

def get_pre_tokens_stat(
    input_path: str,
    special_tokens: list[str]
) -> list[tuple[list[bytes], int]]:
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    pre_tokens = pretokenize(text, special_tokens)

    # Count the frequency of each pretokenized word and store it in a dictionary
    pre_tokens_cnt = {}
    for pt in pre_tokens:
        pre_tokens_cnt[tuple(pt)] = pre_tokens_cnt.get(tuple(pt), 0) + 1

    stat = []
    for pt, cnt in pre_tokens_cnt.items():
        stat.append((pt, cnt))

    return stat

def get_pairs_stat(pre_tokens: list[tuple[list[bytes], int]]) -> dict[tuple[bytes, bytes], tuple[int, dict[int, int]]]:
    pairs_stat = {}

    for i in range(len(pre_tokens)):
        pt_bytes, pt_cnt = pre_tokens[i]
        for j in range(len(pt_bytes) - 1):
            key = (pt_bytes[j], pt_bytes[j+1])
            total, indexes = pairs_stat.get(key, (0, {}))
            total += pt_cnt
            indexes[i] = indexes.get(i, 0) + pt_cnt
            pairs_stat[key] = (total, indexes)

    return pairs_stat

def merge_pair(
    pre_tokens: list[tuple[list[bytes], int]],
    pairs_stat: dict[tuple[bytes, bytes], tuple[int, dict[int, int]]],
    pair: tuple[bytes, bytes],
    stat: tuple[int, dict[int, int]]
    ):

    for i in stat[1].keys():
        pt_bytes, pt_cnt = pre_tokens[i]
        length = len(pt_bytes)
        new_pt_bytes = []

        j = 0
        while j < length:
            t1 = pt_bytes[j - 1] if j > 0 else None
            t2 = pt_bytes[j]
            t3 = pt_bytes[j+1] if j < length - 1 else None
            t4 = pt_bytes[j+2] if j < length - 2 else None

            if t3 and (t2, t3) == pair:
                t2t3 = t2 + t3

                if t1:
                    # Update the count and indexes for the new pair (t1, t2t3)
                    cnt_l, indexes_l = pairs_stat.get((t1, t2t3), (0, {}))
                    cnt_l += pt_cnt
                    indexes_l[i] = indexes_l.get(i, 0) + pt_cnt
                    pairs_stat[(t1, t2t3)] = (cnt_l, indexes_l)

                    # Update the count and indexes for the old pair (t1, t2)
                    if (t1, t2) in pairs_stat:
                        total, indexes = pairs_stat[(t1, t2)]
                        if i in indexes:
                            indexes[i] -= pt_cnt
                            assert indexes[i] >= 0, f"Index {i} has negative count in indexes for pair {(t1, t2)}"
                            if indexes[i] == 0:
                                indexes.pop(i)
                            total -= pt_cnt
                            assert total >= 0, f"Total count for pair {(t1, t2)} is negative after merging, total={total}"
                            pairs_stat[(t1, t2)] = (total, indexes)

                if t4:
                    # Update the count and indexes for the new pair (t2t3, t4)
                    cnt_r, indexes_r = pairs_stat.get((t2t3, t4), (0, {}))
                    cnt_r += pt_cnt
                    indexes_r[i] = indexes_r.get(i, 0) + pt_cnt
                    pairs_stat[(t2t3, t4)] = (cnt_r, indexes_r)

                    # Update the count and indexes for the old pair (t3, t4)
                    if (t3, t4) in pairs_stat:
                        total, indexes = pairs_stat[(t3, t4)]

                        if i in indexes:
                            indexes[i] -= pt_cnt
                            assert indexes[i] >= 0, f"Index {i} has negative count in indexes for pair {(t3, t4)}"
                            if indexes[i] == 0:
                                indexes.pop(i)

                            total -= pt_cnt
                            assert total >= 0, f"Total count for pair {(t3, t4)} is negative after merging, total={total}"
                            pairs_stat[(t3, t4)] = (total, indexes)


                new_pt_bytes.append(t2t3)
                j += 1
            else:
                new_pt_bytes.append(t2)
            j += 1

        pre_tokens[i] = (new_pt_bytes, pt_cnt)

    pairs_stat.pop(pair, None)

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    merges = []

    for i in range(256):
        vocab[i] = bytes([i])

    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    pre_tokens_stat = get_pre_tokens_stat(input_path, special_tokens)
    pairs_stat = get_pairs_stat(pre_tokens_stat)

    while len(vocab) < vocab_size:
        pair, stat = max(pairs_stat.items(), key=lambda item: (item[1][0], item[0]))
        vocab[len(vocab)] = pair[0] + pair[1]
        merges.append((pair[0], pair[1]))

        merge_pair(pre_tokens_stat, pairs_stat, pair, stat)

    return (vocab, merges)