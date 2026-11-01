import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling import ConformerEncoder


class TDTPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def add_sos(self, y):
        B, U, D = y.shape
        start = torch.zeros(B, 1, D, device=y.device)
        y = torch.cat([start, y], dim=1).contiguous()
        return y

    def forward(self, tokens):
        """
        tokens: (B, U)

        return:
            (B, U+1, D)
        """
        if tokens.shape[1] >= 1:
            x = self.embedding(tokens)
            out, _ = self.lstm(x)
            out = self.layer_norm(out)
            out = self.add_sos(out)
        else:
            B = tokens.shape[0]
            out = torch.zeros(B, self.hidden_dim, device=tokens.device)
        return out
    
class TDTJointNet(nn.Module):
    def __init__(self, dim, max_duration, vocab_size):
        super().__init__()

        self.token_head = nn.Linear(dim, vocab_size+1)
        self.duration_head = nn.Linear(dim, max_duration)

        self.fc = nn.Linear(dim, dim)
    
    def forward(self, enc, pred):
        """
        enc: (B, T, D)
        pred: (B, U+1, D)
        """

        enc = enc.unsqueeze(2)
        pred = pred.unsqueeze(1)

        h = torch.tanh(enc + pred)

        h = self.fc(h)

        token_logits = self.token_head(h)
        duration_logits = self.duration_head(h)

        return token_logits, duration_logits

class TDT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_duration: int,
        max_target_length: int = 15,  
        input_dim: tuple = 40,
        encoder_dim: int = 512,
        num_layers: int = 17,
        num_layers_predictor: int = 2,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        super().__init__()

        self.max_target_length = max_target_length
        self.durations = torch.arange(max_duration)
        self.sos_id = 0
        self.blank = 0
        self.vocab_size = vocab_size + 1

        self.encoder = ConformerEncoder(
            input_dim = input_dim,
            encoder_dim = encoder_dim,
            num_layers = num_layers,
            num_attention_heads = num_attention_heads,
            feed_forward_expansion_factor = feed_forward_expansion_factor,
            conv_expansion_factor = conv_expansion_factor,
            input_dropout_p = input_dropout_p,
            feed_forward_dropout_p = feed_forward_dropout_p,
            attention_dropout_p = attention_dropout_p,
            conv_dropout_p = conv_dropout_p,
            conv_kernel_size = conv_kernel_size,
            half_step_residual = half_step_residual,
        )

        self.predictor = TDTPredictor(
            vocab_size=vocab_size+1,
            embed_dim=encoder_dim,
            hidden_dim=encoder_dim,
            num_layers=num_layers_predictor,
        )

        self.joint = TDTJointNet(encoder_dim, max_duration, vocab_size)
    
    def logsumexp(self, a, b):
        ret = torch.logsumexp(torch.stack([a, b]), dim=0)
        return ret
    
    def compute_forward_prob(self, acts, duration_acts, labels, act_lens, label_lens):
        """This function implements Equation 7 in the TDT paper https://arxiv.org/pdf/2304.06795.pdf,
        Simply put, for each alpha(t, u), it sums over the contribution from all incoming blank arcs and non-blank arcs.
        """
        B, T, U, _ = acts.shape

        acts = torch.log_softmax(acts, dim=-1)
        duration_acts = torch.log_softmax(duration_acts, dim=-1)

        log_alpha = torch.zeros(B, T, U, device=acts.device)
        for b in range(B):
            for t in range(T):
                for u in range(U):
                    if u == 0:
                        if t == 0:
                            # both t and u are 0, this is the base case for alphas.
                            log_alpha[b, t, u] = 0.0
                        else:
                            # u = 0 and t != 0: only considers blank emissions.
                            log_alpha[b, t, u] = -1000.0
                            for n, l in enumerate(self.durations):
                                if (
                                    t - l >= 0 and l > 0
                                ):  # checking conditions for blank emission, l has to be at least 1
                                    tmp = (
                                        log_alpha[b, t - l, u]
                                        + acts[b, t - l, u, self.blank]
                                        + duration_acts[b, t - l, u, n]
                                    )
                                    log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

                    else:
                        # u != 0 here, need to consider both blanks and non-blanks.
                        log_alpha[b, t, u] = -1000.0
                        for n, l in enumerate(self.durations):
                            if t - l >= 0:
                                if l > 0:  # for blank emissions. Need to ensure index is not out-of-bound.
                                    tmp = (
                                        log_alpha[b, t - l, u]
                                        + acts[b, t - l, u, self.blank]
                                        + duration_acts[b, t - l, u, n]
                                    )
                                    log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

                                # non-blank emissions.
                                tmp = (
                                    log_alpha[b, t - l, u - 1]
                                    + acts[b, t - l, u - 1, labels[b, u - 1]]
                                    + duration_acts[b, t - l, u - 1, n]
                                )
                                log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

        log_probs = []
        for b in range(B):
            tt = torch.tensor([-1000.0], device=acts.device)

            # need to loop over all possible ways that blank with different durations contributes to the final loss.
            for n, l in enumerate(self.durations):
                if act_lens[b] - l >= 0 and l > 0:
                    bb = (
                        log_alpha[b, act_lens[b] - l, label_lens[b]]
                        + acts[b, act_lens[b] - l, label_lens[b], self.blank]
                        + duration_acts[b, act_lens[b] - l, label_lens[b], n]
                    )

                    tt = self.logsumexp(bb, 1.0 * tt)

            log_probs.append(tt)

        log_prob = torch.stack(log_probs)
        loss = -log_prob.mean()

        return loss, log_alpha

    def tdt_loss(self, acts, duration_acts, labels, act_lens, label_lens):
        """This function implements Equation 7 in the TDT paper https://arxiv.org/pdf/2304.06795.pdf,
        Simply put, for each alpha(t, u), it sums over the contribution from all incoming blank arcs and non-blank arcs.
        """
        B, T, U, _ = acts.shape

        acts = torch.log_softmax(acts, dim=-1)
        duration_acts = torch.log_softmax(duration_acts, dim=-1)

        log_alpha = torch.zeros(B, T, U, device=acts.device)
        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        # both t and u are 0, this is the base case for alphas.
                        log_alpha[:, t, u] = 0.0
                    else:
                        # u = 0 and t != 0: only considers blank emissions.
                        log_alpha[:, t, u] = -1000.0
                        for n, l in enumerate(self.durations):
                            if (
                                t - l >= 0 and l > 0
                            ):  # checking conditions for blank emission, l has to be at least 1
                                tmp = (
                                    log_alpha[:, t - l, u]
                                    + acts[:, t - l, u, self.blank]
                                    + duration_acts[:, t - l, u, n]
                                )
                                log_alpha[:, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[:, t, u])

                else:
                    # u != 0 here, need to consider both blanks and non-blanks.
                    log_alpha[:, t, u] = -1000.0
                    for n, l in enumerate(self.durations):
                        if t - l >= 0:
                            if l > 0:  # for blank emissions. Need to ensure index is not out-of-bound.
                                tmp = (
                                    log_alpha[:, t - l, u]
                                    + acts[:, t - l, u, self.blank]
                                    + duration_acts[:, t - l, u, n]
                                )
                                log_alpha[:, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[:, t, u])

                            # non-blank emissions.
                            tmp = (
                                log_alpha[:, t - l, u - 1]
                                + acts[:, t - l, u - 1].gather(1, labels[:, u - 1].unsqueeze(1)).squeeze(1)
                                + duration_acts[:, t - l, u - 1, n]
                            )
                            log_alpha[:, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[:, t, u])

        log_probs = []
        for b in range(B):
            tt = torch.tensor([-1000.0], device=acts.device)

            # need to loop over all possible ways that blank with different durations contributes to the final loss.
            for n, l in enumerate(self.durations):
                if act_lens[b] - l >= 0 and l > 0:
                    bb = (
                        log_alpha[b, act_lens[b] - l, label_lens[b]]
                        + acts[b, act_lens[b] - l, label_lens[b], self.blank]
                        + duration_acts[b, act_lens[b] - l, label_lens[b], n]
                    )

                    tt = self.logsumexp(bb, 1.0 * tt)

            log_probs.append(tt)

        log_prob = torch.stack(log_probs)
        loss = -log_prob.mean()

        return loss, log_alpha

    def pad_tokens(self, tokens, target_lengths):
        """
        tokens: (L), combined phonemes
        target_lengths: (B, 1)
        """
        
        B = target_lengths.shape[0]

        padded_tokens = torch.zeros(B, self.max_target_length, device=tokens.device, dtype=torch.int64)

        length_cnt = 0
        for b in range(B):
            padded_tokens[b, : target_lengths[b]] = tokens[length_cnt: length_cnt+target_lengths[b]]
            length_cnt += target_lengths[b]

        return padded_tokens

    def forward(self, x, tokens, target_lengths):

        padded_tokens = self.pad_tokens(tokens, target_lengths)

        encoder_out, _ = self.encoder(x)
        predictor_out = self.predictor(padded_tokens)
        token_logits, dur_logits = self.joint(encoder_out, predictor_out)      # (B, T, U+1, V+1), (B, T, U+1, D)

        loss, alpha = self.tdt_loss(
            label_acts=token_logits,
            duration_acts=dur_logits,
            labels=padded_tokens,
            act_lens=torch.ones(x.size(0)) * 26,
            label_lengths=target_lengths,
        )

        # loss, alpha = self.compute_forward_prob(token_logits, dur_logits, padded_tokens, torch.ones(x.size(0), 1, dtype=torch.int64, device=x.device)*26, target_lengths)

        results = {
            "loss": loss,
            "alpha": alpha,
            "token_logits": token_logits,
            "dur_logits": dur_logits,
        }
        return results
    
    def streaming_search(self, target_posteriors, target, target_len, t2skip):

        T, U = target_posteriors.shape[1:3]
        duration = None
        log_alpha = torch.zeros(T, U, device=target_posteriors.device) - 1000
        log_alpha_each_t = [0 for _ in range(T)]

        t = 0
        while t < T:
            for u in range(U):
                if u == 0:
                    if t == 0:
                        log_alpha[t, u] = 0
                    else:
                        log_alpha[t, u] = 0
                else:
                    if t == 0:
                        log_alpha[t, u] = log_alpha[t, u-1] + target_posteriors[t, u-1, target[u-1]]
                    else:
                        from_left = log_alpha[t-duration, u] + target_posteriors[t-duration, u, 0]
                        from_down = log_alpha[t, u-1] + target_posteriors[t, u-1, target[u-1]]
                        if from_left >= from_down:
                            log_alpha[t, u] = from_left
                        else:
                            log_alpha[t, u] = from_down
            out_log_alpha = log_alpha[t, target_len] + target_posteriors[t, target_len, 0]
            log_prob = out_log_alpha.item()
            log_alpha_each_t[t] = out_log_alpha
            duration = t2skip[t]
            t += duration

        return log_prob, log_alpha_each_t

if __name__ == "__main__":
    tdt = TDT(
        vocab_size = 10,
        max_duration = 5,
        max_target_length = 15,  
        input_dim = 40,
    )
    x = torch.randn(1, 201, 40)
    ph = torch.tensor([[1,2,2]])
    t_l = torch.tensor([3])
    tdt(x, ph, t_l)