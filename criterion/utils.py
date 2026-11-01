import torch

import numpy as np
import math
import matplotlib.pyplot as plt

from data.dataset import phonemelm, phoneme_vocab_reverse


def dict_to_device(d, device):
    if isinstance(d, torch.Tensor):
        return d.to(device)
    elif isinstance(d, dict):
        return {k: dict_to_device(v, device) for k, v in d.items()}
    elif isinstance(d, list):
        return [dict_to_device(x, device) for x in d]
    elif isinstance(d, tuple):
        return tuple(dict_to_device(x, device) for x in d)
    else:
        return d
    
def _reconstruct(sequence, blank=0):
    collapsed = []
    prev = None
    for token in sequence:
        if token == blank:
            prev = token
            continue
        if token != prev:
            collapsed.append(token)
        prev = token
    return collapsed

def logsumexp(a, b):
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))

def beam_search_decode(
    log_prob_map,           # (T, B, C)
    beam_size,
    blank=0,
    emission_threshold=np.log(0.01),
    phoneme_vocab_reverse=None,
    phoneme_lm=None,
):
    beam_size = 1
    T, B, C = log_prob_map.shape
    results = []
    all_scores = []
    NEG_INF = -float("inf")

    for b in range(B):
        beams = {(): (0.0, NEG_INF)}  # prefix -> (p_blank, p_nonblank)

        for t in range(T):
            next_beams = {}

            probs = log_prob_map[t, b]

            for prefix, (p_b, p_nb) in beams.items():
                for c in range(C):
                    p = probs[c].item()
                    if p < emission_threshold:
                        continue

                    if c == blank:
                        # blank keeps prefix
                        nb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                        # print(nb)
                        next_beams[prefix] = (
                            logsumexp(nb[0], p_b + p),
                            logsumexp(nb[1], p_nb + p),
                        )
                    else:
                        new_prefix = prefix + (c,)
                        nb = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                        # print(nb)
                        if prefix and prefix[-1] == c:
                            # repeat token
                            # next_beams[new_prefix] = (
                            #     nb[0],
                            #     logsumexp(nb[1], p_b + p),
                            # )
                            nb_same = next_beams.get(prefix, (NEG_INF, NEG_INF))
                            next_beams[prefix] = (
                                nb_same[0],
                                logsumexp(nb_same[1], p_nb + p),
                            )

                            # 情況 2：經過 blank 後再次輸出相同 token，增加輸出長度
                            new_prefix = prefix + (c,)
                            nb_new = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                            next_beams[new_prefix] = (
                                nb_new[0],
                                logsumexp(nb_new[1], p_b + p),
                            )
                        else:
                            next_beams[new_prefix] = (
                                nb[0],
                                logsumexp(nb[1], logsumexp(p_b, p_nb) + p)
                            )

            # prune
            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda x: logsumexp(x[1][0], x[1][1]),
                    reverse=True,
                )[:beam_size]
            )

        # LM rescoring
        scored = []
        for prefix, (p_b, p_nb) in beams.items():
            score = logsumexp(p_b, p_nb)
            if phoneme_lm is not None:
                tokens = [phoneme_vocab_reverse[p] for p in prefix]
                score += phoneme_lm.scoring(tokens)
            scored.append((list(prefix), score))

            if score < -100:
                plt.imshow(log_prob_map[:, 0].detach().cpu().numpy().T)
                plt.show()


        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            best_tokens, best_score = scored[0]
        else:
            best_tokens, best_score = [], NEG_INF

        results.append(_reconstruct(best_tokens))
        all_scores.append(best_score)

    return results, all_scores