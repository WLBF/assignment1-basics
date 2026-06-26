corpus = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

vocab = ['<|endoftext|>'] + [bytes([i]) for i in range(256)]

pre_tokens = {}

for token in [bytes(word, 'utf-8') for word in corpus.split()]:
    key = tuple(bytes([b]) for b in token)
    pre_tokens[key] = pre_tokens.get(key, 0) + 1

print(pre_tokens)

for _ in range(6):
    print('===================')
    pairs = {}
    for pt, cnt in pre_tokens.items():
        pt_bytes = list(pt)
        for i in range(len(pt_bytes) - 1):
            p = (pt_bytes[i], pt_bytes[i+1])
            pairs[p] = pairs.get(p, 0) + cnt

    print(pairs)
    max_pair, max_cnt = max(pairs.items(), key=lambda item: (item[1], item[0]))
    # print(max_pair, max_cnt)    
    vocab.append(max_pair[0] + max_pair[1])

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

    print('-----------------------------')
    # print(pre_tokens)

print(len(vocab), vocab)