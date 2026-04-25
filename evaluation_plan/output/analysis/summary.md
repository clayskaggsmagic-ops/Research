# Experiment Results Summary

Scorable questions: 103
Common qids across all 7 experiments (for fair comparison): 103

## Record counts and errors
| exp | label | total | ok | errors | tok_in | tok_out |
|---|---|---:|---:|---:|---:|---:|
| e1 | Trump × CHRONOS broad-15 | 515 | 515 | 0 | 12,673,555 | 74,802 |
| e1p | Trump × CHRONOS broad-8 (compressed) | 515 | 515 | 0 | 7,523,555 | 73,724 |
| e2 | Trump × CHRONOS refined (2-stage) | 515 | 515 | 0 | 1,525,011 | 69,481 |
| e3 | Trump × no briefing | 515 | 515 | 0 | 668,030 | 67,869 |
| e4 | Analyst × CHRONOS broad-15 | 515 | 515 | 0 | 12,772,424 | 88,651 |
| e5 | Analyst × Tavily web search (date-bounded, strict) | 515 | 515 | 0 | 723,336 | 77,675 |
| e6 | Analyst × Tavily web search (answerability gate, unbounded) | 515 | 515 | 0 | 1,279,478 | 83,743 |

## Complete-case analysis (per-exp, each experiment's own surviving qids)
| exp | n_qids | mean Brier | mean p(correct) |
|---|---:|---:|---:|
| e1 | 103 | 0.4774 | 0.4276 |
| e1p | 103 | 0.4835 | 0.4219 |
| e2 | 103 | 0.4723 | 0.4296 |
| e3 | 103 | 0.4804 | 0.4171 |
| e4 | 103 | 0.4808 | 0.4254 |
| e5 | 103 | 0.4863 | 0.4054 |
| e6 | 103 | 0.4518 | 0.4609 |

## Fair comparison (restricted to common qids across all 7 experiments)
| exp | n_qids | mean Brier | mean p(correct) |
|---|---:|---:|---:|
| e1 | 103 | 0.4774 | 0.4276 |
| e1p | 103 | 0.4835 | 0.4219 |
| e2 | 103 | 0.4723 | 0.4296 |
| e3 | 103 | 0.4804 | 0.4171 |
| e4 | 103 | 0.4808 | 0.4254 |
| e5 | 103 | 0.4863 | 0.4054 |
| e6 | 103 | 0.4518 | 0.4609 |

## By question format (common qids only)
| exp | binary n | binary Brier | action n | action Brier |
|---|---:|---:|---:|---:|
| e1 | 58 | 0.2457 | 45 | 0.7762 |
| e1p | 58 | 0.2523 | 45 | 0.7814 |
| e2 | 58 | 0.2717 | 45 | 0.7310 |
| e3 | 58 | 0.2779 | 45 | 0.7413 |
| e4 | 58 | 0.2370 | 45 | 0.7952 |
| e5 | 58 | 0.2474 | 45 | 0.7943 |
| e6 | 58 | 0.2032 | 45 | 0.7723 |

## By difficulty (common qids only)
| exp | hard (n, Brier) | medium (n, Brier) |
|---|---|---|
| e1 | 71, 0.3395 | 32, 0.7834 |
| e1p | 71, 0.3446 | 32, 0.7917 |
| e2 | 71, 0.3488 | 32, 0.7464 |
| e3 | 71, 0.3616 | 32, 0.7439 |
| e4 | 71, 0.3420 | 32, 0.7890 |
| e5 | 71, 0.3466 | 32, 0.7964 |
| e6 | 71, 0.3164 | 32, 0.7523 |

## By domain (common qids only)
| exp | executive_orders | foreign_policy | legal_judicial | legislative | personnel | public_comms | trade_tariffs |
|---|---|---|---|---|---|---|---|
| e1 | n=19, 0.5075 | n=18, 0.4445 | n=8, 0.4863 | n=11, 0.5509 | n=14, 0.4245 | n=9, 0.5013 | n=24, 0.4637 |
| e1p | n=19, 0.5207 | n=18, 0.4472 | n=8, 0.4974 | n=11, 0.5430 | n=14, 0.4293 | n=9, 0.5291 | n=24, 0.4638 |
| e2 | n=19, 0.5236 | n=18, 0.4381 | n=8, 0.5419 | n=11, 0.5199 | n=14, 0.3984 | n=9, 0.4444 | n=24, 0.4661 |
| e3 | n=19, 0.4777 | n=18, 0.4882 | n=8, 0.4748 | n=11, 0.5430 | n=14, 0.4707 | n=9, 0.4199 | n=24, 0.4781 |
| e4 | n=19, 0.5555 | n=18, 0.4564 | n=8, 0.3914 | n=11, 0.5415 | n=14, 0.4590 | n=9, 0.5375 | n=24, 0.4335 |
| e5 | n=19, 0.5532 | n=18, 0.4474 | n=8, 0.4070 | n=11, 0.5123 | n=14, 0.4394 | n=9, 0.5664 | n=24, 0.4745 |
| e6 | n=19, 0.4999 | n=18, 0.4434 | n=8, 0.3018 | n=11, 0.5185 | n=14, 0.3935 | n=9, 0.5655 | n=24, 0.4311 |

## Paired deltas (per-question Brier difference, common qids)
Positive mean_delta means `a` is WORSE than `b` on that axis.
| contrast | mean Δ | median Δ | a-better | b-better | ties | n |
|---|---:|---:|---:|---:|---:|---:|
| persona (e1 vs e4, same briefing) | -0.0034 | +0.0000 | 50 | 46 | 7 | 103 |
| briefing_vs_none (e1 vs e3) | -0.0029 | +0.0000 | 45 | 48 | 10 | 103 |
| refinement (e1 vs e2) | +0.0051 | +0.0000 | 41 | 51 | 11 | 103 |
| compression (e1 vs e1p) | -0.0060 | +0.0000 | 47 | 40 | 16 | 103 |
| web_vs_curated (e4 vs e5) | -0.0055 | +0.0000 | 51 | 45 | 7 | 103 |
| trump_persona_on_web (e5 vs e3) | +0.0060 | +0.0035 | 46 | 54 | 3 | 103 |
| gate_vs_bounded_web (e6 vs e5) | -0.0345 | -0.0240 | 60 | 41 | 2 | 103 |
| gate_vs_no_briefing (e6 vs e3) | -0.0285 | -0.0206 | 59 | 38 | 6 | 103 |
| gate_vs_best_chronos (e6 vs e2) | -0.0205 | -0.0160 | 54 | 44 | 5 | 103 |

## Sample-level variance (mean intra-question stdev of p(correct))
| exp | mean stdev | max stdev |
|---|---:|---:|
| e1 | 0.0321 | 0.1356 |
| e1p | 0.0382 | 0.1470 |
| e2 | 0.0258 | 0.1020 |
| e3 | 0.0251 | 0.1020 |
| e4 | 0.0293 | 0.1833 |
| e5 | 0.0228 | 0.0980 |
| e6 | 0.0325 | 0.3923 |

## E5 attrition
Questions with zero valid E5 samples: 0
