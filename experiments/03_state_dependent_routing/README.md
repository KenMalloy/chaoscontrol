# Experiment 03: State-Dependent Routing

## Hypothesis
B(x,h) outperforms B(x); distributed topology outperforms monolithic NN.

## Null hypothesis
No rich B variant beats the no-rich-B baseline with the same A mode.

## Predictions
- Any rich B mode outperforms none (state-dependent routing helps)
- Hub and assembly topologies outperform monolithic NN
- Assembly with 4 settling steps outperforms assembly with 2
- The interaction between A mode and rich B is additive, not multiplicative
- full_assembly_2 is the overall winner

## Method
Ten configs: crossing A mode (diag, full) with rich B mode (none, nn, hub, assembly, hybrid).
Assembly configs include settling steps. All at model_dim=128, num_layers=4. Fixed 300s budget.
Key question: does distributed beat monolithic NN? If not, the brain-inspired topology doesn't matter.

## Dependencies
01, 02 (for baseline comparison and A mode interpretation).

## Kill criteria
If no rich B variant beats the none baseline, state-dependent routing adds complexity without benefit.
