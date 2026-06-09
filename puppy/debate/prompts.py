DEBATE_QUESTION_TEMPLATE = (
    "Graph risk conflicts with the local decision for {symbol}. "
    "Local decision={local_decision}, text_signal={text_signal}, "
    "graph_signal={graph_signal}, fusion_score={fusion_score:.3f}. "
    "Justify the local decision using cited memory ids."
)

DEBATE_JUDGE_POLICY = (
    "Return one of keep, reduce, override_hold, or override_sell. "
    "Prefer rule-based safety when graph risk contradicts a buy signal."
)
