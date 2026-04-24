# Experiment Results Summary

Scorable questions: 103
Common qids across all 6 experiments (for fair comparison): 88

## Record counts and errors
| exp | label | total | ok | errors | tok_in | tok_out |
|---|---|---:|---:|---:|---:|---:|
| e1 | Trump × CHRONOS broad-15 | 515 | 515 | 0 | 10,317,240 | 73,010 |
| e1p | Trump × CHRONOS broad-8 (compressed) | 515 | 515 | 0 | 6,208,970 | 71,542 |
| e2 | Trump × CHRONOS refined (2-stage) | 515 | 515 | 0 | 1,252,595 | 70,956 |
| e3 | Trump × no briefing | 515 | 515 | 0 | 659,050 | 69,389 |
| e4 | Analyst × CHRONOS broad-15 | 515 | 514 | 1 | 11,373,489 | 99,678 |
| e5 | Analyst × Tavily web search | 515 | 392 | 123 | 2,033,852 | 81,285 |

## Complete-case analysis (per-exp, each experiment's own surviving qids)
| exp | n_qids | mean Brier | mean p(correct) |
|---|---:|---:|---:|
| e1 | 103 | 0.5172 | 0.4621 |
| e1p | 103 | 0.5367 | 0.4464 |
| e2 | 103 | 0.5187 | 0.4584 |
| e3 | 103 | 0.5356 | 0.4413 |
| e4 | 103 | 0.4807 | 0.4705 |
| e5 | 88 | 0.4409 | 0.5750 |

## Fair comparison (restricted to common qids across all 6 experiments)
| exp | n_qids | mean Brier | mean p(correct) |
|---|---:|---:|---:|
| e1 | 88 | 0.5114 | 0.4669 |
| e1p | 88 | 0.5332 | 0.4497 |
| e2 | 88 | 0.5208 | 0.4579 |
| e3 | 88 | 0.5377 | 0.4423 |
| e4 | 88 | 0.4830 | 0.4681 |
| e5 | 88 | 0.4409 | 0.5750 |

## By question format (common qids only)
| exp | binary n | binary Brier | action n | action Brier |
|---|---:|---:|---:|---:|
| e1 | 50 | 0.3139 | 38 | 0.7712 |
| e1p | 50 | 0.3342 | 38 | 0.7951 |
| e2 | 50 | 0.3255 | 38 | 0.7777 |
| e3 | 50 | 0.3231 | 38 | 0.8200 |
| e4 | 50 | 0.2774 | 38 | 0.7535 |
| e5 | 50 | 0.2191 | 38 | 0.7326 |

## By difficulty (common qids only)
| exp | hard (n, Brier) | medium (n, Brier) |
|---|---|---|
| e1 | 61, 0.4116 | 27, 0.7369 |
| e1p | 61, 0.4228 | 27, 0.7828 |
| e2 | 61, 0.4078 | 27, 0.7761 |
| e3 | 61, 0.4124 | 27, 0.8208 |
| e4 | 61, 0.3698 | 27, 0.7387 |
| e5 | 61, 0.3171 | 27, 0.7205 |

## By domain (common qids only)
| exp | executive_orders | foreign_policy | legal_judicial | legislative | personnel | public_comms | trade_tariffs |
|---|---|---|---|---|---|---|---|
| e1 | n=18, 0.5655 | n=12, 0.6280 | n=8, 0.5089 | n=11, 0.4781 | n=6, 0.4057 | n=9, 0.3831 | n=24, 0.5031 |
| e1p | n=18, 0.5786 | n=12, 0.6027 | n=8, 0.5402 | n=11, 0.5466 | n=6, 0.4320 | n=9, 0.3874 | n=24, 0.5360 |
| e2 | n=18, 0.5838 | n=12, 0.6315 | n=8, 0.5459 | n=11, 0.5138 | n=6, 0.3902 | n=9, 0.3494 | n=24, 0.5099 |
| e3 | n=18, 0.5651 | n=12, 0.6323 | n=8, 0.5052 | n=11, 0.5317 | n=6, 0.4920 | n=9, 0.3396 | n=24, 0.5691 |
| e4 | n=18, 0.5009 | n=12, 0.5100 | n=8, 0.4777 | n=11, 0.4422 | n=6, 0.4624 | n=9, 0.5085 | n=24, 0.4720 |
| e5 | n=18, 0.4839 | n=12, 0.5156 | n=8, 0.2499 | n=11, 0.4492 | n=6, 0.3075 | n=9, 0.5315 | n=24, 0.4304 |

## Paired deltas (per-question Brier difference, common qids)
Positive mean_delta means `a` is WORSE than `b` on that axis.
| contrast | mean Δ | median Δ | a-better | b-better | ties | n |
|---|---:|---:|---:|---:|---:|---:|
| persona (e1 vs e4, same briefing) | +0.0284 | +0.0055 | 41 | 47 | 0 | 88 |
| briefing_vs_none (e1 vs e3) | -0.0263 | -0.0015 | 47 | 39 | 2 | 88 |
| refinement (e1 vs e2) | -0.0094 | +0.0010 | 40 | 45 | 3 | 88 |
| compression (e1 vs e1p) | -0.0218 | +0.0000 | 40 | 43 | 5 | 88 |
| web_vs_curated (e4 vs e5) | +0.0421 | +0.0307 | 37 | 51 | 0 | 88 |
| trump_persona_on_web (e5 vs e3) | -0.0968 | -0.0235 | 56 | 32 | 0 | 88 |

## Sample-level variance (mean intra-question stdev of p(correct))
| exp | mean stdev | max stdev |
|---|---:|---:|
| e1 | 0.0679 | 0.2713 |
| e1p | 0.0579 | 0.2646 |
| e2 | 0.0642 | 0.3370 |
| e3 | 0.0624 | 0.2874 |
| e4 | 0.0860 | 0.3108 |
| e5 | 0.1273 | 0.4243 |

## E5 attrition
Questions with zero valid E5 samples: 15
  Q-S-026-03, Q-S-032-02, Q-S-034-03, Q-S-035-01, Q-S-036-01, Q-S-037-01, Q-S-037-03, Q-S-038-01, Q-S-038-02, Q-S-039-01, Q-S-039-02, Q-S-040-01, Q-S-040-02, Q-S-041-01, Q-S-041-02
