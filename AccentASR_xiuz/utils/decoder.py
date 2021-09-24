import torch


def greed_decode(log_prob):
    # TODO:shrink chars
    ctc_pred = torch.argmax(log_prob, dim=-1).cpu()
    return ctc_pred.cpu()
