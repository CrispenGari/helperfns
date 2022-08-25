import torch

def text_pipeline(x, tokenizer, vocab, unk_token):
    values = list()
    tokens = tokenizer(x)
    for token in tokens:
        try:
            v = vocab[token]
        except RuntimeError as e:
            v = vocab[unk_token]
            values.append(v)
    return values

def tokenize_batch(batch, max_len=50, padding="pre"):
    assert padding=="pre" or padding=="post", "the padding can be either pre or post"
    labels_list, text_list = [], []
    for _label, _text in batch:
        labels_list.append(label_pipeline(_label))
        text_holder = torch.zeros(max_len, dtype=torch.int32) # fixed size tensor of max_len with <pad> = 0
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int32)
        pos = min(max_len, len(processed_text))
        if padding == "pre":
            text_holder[:pos] = processed_text[:pos]
        else:
            text_holder[-pos:] = processed_text[-pos:]
        text_list.append(text_holder.unsqueeze(dim=0))
    return torch.LongTensor(labels_list), torch.cat(text_list, dim=0)

labels_dict = {l:i for (i, l) in enumerate({'af', 'en', 'st', 'ts', 'xh', 'zu'})}

def label_pipeline(x:str)->int:
  return ['af', 'en', 'st', 'ts', 'xh', 'zu'].index(x)