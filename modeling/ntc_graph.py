import numpy as np


NEGINF          = -1e30
MAX_STATES      = 16
TOTAL_LABELS    = 38
WILDCARD_LABEL  = 37
BLANK           = 0
ARC_TYPES = {
    "normal-transition": 0,
    "self-loop": 1,
    "bypass": 2
}

class NTCArc():
    def __init__(self, dst_state, token_id, type, weight):
        self.dst_state = dst_state
        self.token_id = token_id
        self.type = type
        self.weight = weight

class NTCState():
    def __init__(self, first_arc, arc_count, keyword_id, is_final, state_id):
        self.first_arc = first_arc
        self.arc_count = arc_count
        self.keyword_id = keyword_id
        self.is_final = is_final
        self.state_id = state_id

class NTCGraph():
    def __init__(self, keyword_id, normal_weight, selfloop_weight, bypass_weight):
        self.num_states = 0
        self.num_arcs = 0

        self.normal_weight = normal_weight
        self.selfloop_weight = selfloop_weight
        self.bypass_weight = bypass_weight

        self.states = []
        self.arcs = []

        self.keyword_id = keyword_id
        self.start_state = None
        self.final_state= None

    def graph_add_state(self, is_final):
        assert self.num_states <= MAX_STATES, "Max num states exceed"

        state_id = self.num_states
        self.states.append(
            NTCState(0, 0, self.keyword_id, is_final, state_id)
        )
        self.num_states += 1

    def graph_add_arc(self, src_state_id, dst_state_id, type, token_id, weight):
        src_state = self.states[src_state_id]

        if src_state.arc_count == 0:
            src_state.first_arc = self.num_arcs
        
        assert src_state.first_arc + src_state.arc_count == self.num_arcs,\
        "incontinuous arcs"

        self.arcs.append(NTCArc(dst_state_id, token_id, type, weight))

        src_state.arc_count += 1
        self.num_arcs += 1

    def graph_add_keyword(self, tokens):
        for i in range(len(tokens) + 1):
            self.graph_add_state(i==len(tokens))

        self.start_state = self.states[0]
        self.final_state = self.states[-1]

        for i in range(len(tokens) + 1):
            self.graph_add_arc(i, i, ARC_TYPES["self-loop"], 0, self.selfloop_weight)

            if i < len(tokens):
                self.graph_add_arc(i, i + 1, ARC_TYPES["normal-transition"], tokens[i], self.normal_weight)

            if i < len(tokens) - 1:
                self.graph_add_arc(i, i + 2, ARC_TYPES["bypass"], 0, self.bypass_weight)

    def wildcard_log_prob(self, frame):
        return np.mean(frame[1:])

    def update_max(self, dst_lp, candidate):
        if candidate > dst_lp:
            dst_lp = candidate
        return dst_lp

    def graph_decode(self, log_probs, threshold):

        t, c = log_probs.shape

        workspace = np.ones((2, self.num_states, TOTAL_LABELS, 2), dtype=np.float32) * -1e30
        workspace[0, 0, BLANK, 0] = 0.0

        for i in range(t):
            log_prob = log_probs[i]
            wc_lp = self.wildcard_log_prob(log_prob)
            workspace[1] = NEGINF

            for s in range(self.num_states):
                state = self.states[s]

                for l in range(TOTAL_LABELS):
                    for used in range(2):
                        base = workspace[0, s, l, used]
                        
                        if (base <= NEGINF / 2.0):
                            continue

                        # blank emission
                        workspace[1, s, BLANK, used] = self.update_max(workspace[1, s, BLANK, used], base + log_prob[BLANK])

                        # repeated token
                        if l >= 1 and l < c:
                            workspace[1, s, l, used] = self.update_max(workspace[1, s, l, used], base + log_prob[l])

                        # outgoing arcs
                        arc_end = state.first_arc + state.arc_count
                        for a in range(arc_end):
                            arc = self.arcs[a]
                            assert arc.dst_state < self.num_states
                            next_used = used
                            if arc.type == ARC_TYPES["normal-transition"]:
                                token = arc.token_id
                                assert (token != BLANK and token < c), "invalid arc token"

                                # skip if token is same as last token, it must has passed through blank
                                if l == token:
                                    continue

                                candidate = base + log_prob[token] + arc.weight

                                next_label = token
                                next_used = 1
                            elif arc.type == ARC_TYPES["self-loop"] or arc.type == ARC_TYPES["bypass"]:
                                candidate = base + wc_lp + arc.weight
                                next_label = WILDCARD_LABEL
                            else:
                                assert "undefined arc"

                            workspace[1, arc.dst_state, next_label, next_used] = self.update_max(workspace[1, arc.dst_state, next_label, next_used], candidate)
            
            workspace = workspace[::-1]

        best_score = NEGINF
        best_keyword = 0
        for s in range(self.num_states):
            state = self.states[s]
            if not state.is_final:
                continue
            for l in range(TOTAL_LABELS):
                score = workspace[0, s, l, 1]
                if score > best_score:
                    best_score = score
                    best_keyword = state.keyword_id
        
        accepted = 1 if best_score >= threshold else 0

        result = {
            "keyword_id": best_keyword,
            "best_score": best_score,
            "accepted": accepted
        }
        return result
        
if __name__ == "__main__":
    graph = NTCGraph(1, 0, 0, 0)
    tokens = [22, 5, 13]
    log_probs = np.random.randn(21, 37)
    graph.graph_add_keyword(tokens)
    result = graph.graph_decode(log_probs, 1)
    print(result)