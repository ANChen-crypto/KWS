from collections import defaultdict, Counter
import math


class PhonemeLM:
    def __init__(self, sequences, smoothing=1e-8):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()

        self.vocab_size = 0
        self.smoothing = smoothing

        for seq in sequences:
            prev = None
            for p in seq:
                self.unigram_counts[p] += 1
                if prev is not None:
                    self.bigram_counts[prev][p] += 1
                prev = p
        self.vocab_size = len(self.unigram_counts)

    def scoring(self, seq):
        log_prob = 0.0
        prev = None
        for p in seq:
            if prev is None:
                prob = (self.unigram_counts[p] + self.smoothing) / (sum(self.unigram_counts.values()) + self.smoothing*self.vocab_size)
            else:
                prob = (self.bigram_counts[prev][p] + self.smoothing) / (sum(self.bigram_counts[prev].values()) + self.smoothing*self.vocab_size)
            prev = p
            log_prob += math.log(prob)
        return log_prob