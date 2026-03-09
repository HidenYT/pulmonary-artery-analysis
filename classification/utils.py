import torch

def keep_longest_run(y_bin):
    y = y_bin.int()

    max_len = 0
    max_start = -1

    current_len = 0
    current_start = 0

    for i in range(len(y)):
        if y[i] == 1:
            if current_len == 0:
                current_start = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
            current_len = 0

    if current_len > max_len:
        max_len = current_len
        max_start = current_start

    result = torch.zeros_like(y)
    if max_len > 0:
        result[max_start:max_start + max_len] = 1

    return result
