# SemanticEngine SSM: A Biologically-Inspired State-Space Model with Typed Routing and Structured Memory Consolidation

## Paper Skeleton

**Venue target:** TMLR / JMLR (journal-length, 15-20 pages)

**One-sentence contribution:** A state-space model with Wernicke-style typed routing and sleep-cycle memory consolidation achieves [X] bpb on FineWeb, with ablations isolating the contribution of each biological mechanism.

**Framing:** Engineering paper. Architecture + ablations + numbers. Biological motivation colors design choices but doesn't replace evidence. Results from H100 final architecture; A40 development experiments inform the design in section 3.

---

## 1. Introduction (1.5 pages)

- The problem: parameter golf — maximum compression quality in 16MB
- Current approaches scale dense transformers. We take a different path: an SSM with a semantic engine that organizes what it learns, not just predicts harder
- One paragraph in Ken's voice about the origin: wanted to use all the hardware, not just attention heads. SSMs gave a reason to come back to AI
- The contribution: full architecture with ablation cascade proving each component earns its place
- What we found: [summary of key numbers]

## 2. Architecture (4-5 pages)

Present the full system top-down. Each subsection: what it does, why it's there (one sentence of biological motivation), how it connects to the next component. Enough detail to reimplement.

- **2.1 SSM Core** — State-space recurrence, diag A-mode, criticality regularization. How it relates to Mamba-2
- **2.2 Wernicke Typed Routing** — VQ/MoE routing into k_max buckets. Per-bucket expert projections (bottleneck for scaling). Balance loss
- **2.3 Episodic Memory** — Multi-slot outer model, cue-dependent retrieval, typed storage (bucket-tagged slots), compression with survival scoring
- **2.4 Semantic Tier** — CONDITIONAL: include only if experiment 13 shows it helps. Slow-moving bases extracted from episodes
- **2.5 Sleep Consolidation** — N1/N2/N3/REM stages. Fixed-interval triggering. The claim: sleep costs training steps but produces better-organized memory
- **2.6 Bucket Affinity Matrix** — CONDITIONAL: include only if tested and helps. Learned cross-type merge compatibility, joint ranking by sim x affinity, emergent cluster discovery
- **2.7 Artifact Serialization** — int6 quantization, LZMA compression, auto-escalation to fit 16MB. What goes in the artifact, what doesn't (sleep is training-only)

Diagrams: full pipeline, sleep cycle, merge ranking

## 3. Experimental Setup (2 pages)

- **3.1 Dataset and Metric** — FineWeb, raw bytes (vocab=256), bits-per-byte. Competition context (baseline 1.22, SOTA 1.11)
- **3.2 Training Protocol** — Budget, hardware, optimizer, lr, batch size. A40 development → H100 final runs. All numbers in section 4 from H100 final architecture
- **3.3 Ablation Cascade Design** — Layered approach: bare SSM → +Wernicke → +memory → +sleep → full. Each layer tested independently before stacking. Constants validated via 1-D sweeps (section 4.3)
- **3.4 Statistical Methodology** — 7 seeds minimum, paired Wilcoxon signed-rank, Holm-Bonferroni correction for confirmatory families, bootstrap 95% CIs, Cohen's d. Confirmatory/exploratory split. Post-selected winners require 8+ seed confirmation

## 4. Results (3-4 pages)

All results from H100 final architecture.

- **4.1 Architecture Ablation Cascade** — Main result table. Mamba-2 (external baseline) → bare SSM → +Wernicke → +memory → +sleep → full architecture. Each row adds one component. Delta-bpb per component with CIs

- **4.2 Sleep Stage Ablation** — The 9-condition breakdown from experiment 11. no_sleep, n3_only, n2_n3, n2_n3_rem_base, n2_n3_rem_validate, n2_n3_rem_cfr, n2_n3_rem_reactivate, n2_n3_rem_all, full_cycle. Which stages earn their keep

- **4.3 Constants Sensitivity** — Five sweeps from experiment 13:
  - Criticality target (0.80-0.96) on bare SSM
  - Memory slot dimension (32/64/128)
  - Max slots (32/64/128)
  - Semantic tier (off vs bases x rate grid)
  - Merge similarity threshold (0.75-0.95)
  - For each: sweep curve with CIs, whether default is optimal or should change

- **4.4 Bucket Count (k_max) Sweep** — 16/32/64 at matched params (expert bottleneck). Wernicke-only and full-stack variants. "How many types does byte-level English have?"

- **4.5 Artifact Compression** — Three-number eval: bpb_pretrain → bpb_artifact (int6) → bpb_ttt. Quantization degradation. Artifact size breakdown. Competition placement

## 5. Analysis (2-3 pages)

- **5.1 Where the bpb Comes From** — Decomposition of final number by component. Stacked bar or pie chart. Which biological mechanism pulls the most weight
- **5.2 The Memory Crossover** — At short budgets memory hurts, at long budgets it helps. Plot bpb vs budget with/without memory. The crossover point as a finding
- **5.3 What Sleep Consolidation Changes** — Diagnostic: slot survival distributions before/after sleep, compression rates, affinity matrix structure (if applicable). Not just bpb — what happens inside
- **5.4 Bucket Utilization** — What the routing learned. Are buckets equally used or specialized? Can we characterize the clusters? The "types of English at byte level" finding

## 6. Related Work (2 pages)

- **6.1 State-Space Models** — S4 (Gu et al. 2022), Mamba (Gu & Dao 2023), Mamba-2 (Dao & Gu 2024). We build on this lineage. Our contribution is the semantic engine on top, not the recurrence itself

- **6.2 Memory-Augmented Models** — NTMs, memory networks, Memorizing Transformers (Wu et al. 2022), kNN-LM (Khandelwal et al. 2020), Larimar (IBM 2024). These augment models with external memory stores but lack consolidation mechanisms. Our approach follows complementary learning systems (McClelland et al. 1995; updated by Kumaran et al. 2016) — fast hippocampal encoding + slow neocortical consolidation — rather than the differentiable-memory tradition

- **6.3 Mixture of Experts** — GShard, Switch Transformer. Wernicke is MoE for typed semantic routing, not capacity scaling. Buckets organize memory, not just compute

- **6.4 Sleep and Consolidation in Neural Networks** — The most important positioning section. Three threads:

  *Foundational:* The wake-sleep algorithm (Hinton et al. 1995) alternates generative and recognition phases but does not consolidate memory. EWC (Kirkpatrick et al. 2017) constrains weights during new learning, inspired by synaptic consolidation but operating in weight space, not memory space.

  *Sleep-cycle approaches in vision:* Robinson et al. (2022) model NREM, REM, and synaptic downscaling for continual learning on CIFAR-100 — the first multi-stage sleep framework in deep learning. WSCL (Hopf et al. 2024) implements wake/NREM/REM phases for image classification with replay from short-term and long-term memory. Tadros et al. (2022, Nature Comms) show sleep-like unsupervised replay reduces catastrophic forgetting. SIESTA (Harun et al. 2023) uses wake/sleep for on-device continual learning. **All target image classification and replay stored samples rather than maintaining discrete episodic memory slots.**

  *Sleep-inspired approaches in language:* SleepGate (2026) augments transformer LLMs with learned tag→prune→merge over the KV cache — strikingly similar mechanics to our N2→N3 pipeline, but operates at inference time on the attention cache, not during training on an episodic memory store. "Language Models Need Sleep" (OpenReview 2024) uses "sleep" for knowledge distillation and self-generated training data — same domain and terminology, entirely different mechanism.

  *The gap we fill:* Hayes et al. (2021, Neural Computation) survey replay in deep learning and explicitly identify missing biological elements: structured sleep stages, surprise-modulated replay, and NREM/REM differentiation. Our work addresses this gap directly. To our knowledge, no prior work maintains typed discrete episodic memory slots and applies structured N1→N2→N3→REM consolidation (utility scoring, typed pruning/merging, dream-based validation) during language model training. The closest structural analogues either target images (WSCL, Robinson et al.) or operate at inference time (SleepGate).

## 7. Discussion (2 pages)

- **7.1 What Worked and What Didn't** — Honest accounting per component
- **7.2 Limitations** — Small model (128d/4L). Short budget. A40→H100 transfer assumptions. 1-D sweeps, not full grids. Synergy matrix and polyphasic sleep implemented but untested at scale
- **7.3 The Scaling Question** — Does the semantic engine's value grow with budget and params? The memory crossover suggests yes, but not proven beyond 600s/128d
- **7.4 Biological Correspondence** — We borrowed principles, not mechanisms. Wernicke buckets aren't Wernicke's area. Sleep N2 isn't sleep spindles. The biology motivated the design; the ablations validate it

## 8. Conclusion (0.5 page)

- What we built, what it achieves, the number
- Two things worth remembering: (1) biological principles led to engineering choices that each earned their bpb, (2) structured memory consolidation is an untapped direction for sequence models

---

## Supplementary Material

- Per-seed result tables for all experiments (reproducibility)
- Full training curves
- Hyperparameter configurations (YAML dumps)

---

## Figures and Tables Needed

| Figure/Table | Section | Status |
|---|---|---|
| Architecture diagram (full pipeline) | 2 | Can draw now |
| Sleep cycle diagram (N1→N2→N3→REM) | 2.5 | Can draw now |
| Merge ranking illustration (sim × affinity) | 2.6 | Can draw now (conditional) |
| Main ablation cascade table | 4.1 | Needs H100 data |
| Sleep stage ablation table | 4.2 | Needs exp 11 data |
| Constants sensitivity plots (5 sweep curves) | 4.3 | Needs exp 13 data |
| k_max sweep plot | 4.4 | Needs baseline data |
| Three-number artifact eval | 4.5 | Needs exp 10b data |
| bpb decomposition (stacked bar) | 5.1 | Needs final data |
| Memory crossover plot (bpb vs budget) | 5.2 | Needs multi-budget runs |
| Bucket utilization heatmap | 5.4 | Needs trained model inspection |

## What We Can Write Now

- Section 1 (introduction — except final numbers)
- Section 2 (architecture — all of it)
- Section 3 (experimental setup — all of it)
- Section 6 (related work — all of it)
- Section 7.2-7.4 (limitations, scaling question, biological correspondence)
- Section 8 (conclusion — except numbers)

## What We Need Data For

- Section 4 (all results)
- Section 5 (all analysis)
- Section 7.1 (what worked)
- Final numbers in sections 1 and 8

## References

**Neuroscience / biological basis:**
- McClelland, McNaughton & O'Reilly 1995 — Complementary learning systems (*Psych Review*)
- Kumaran, Hassabis & McClelland 2016 — CLS updated (*Trends Cog Sci*)
- Doya 1999 — Cerebellum/cortex/basal ganglia computational trichotomy (*Neural Networks*)
- Herculano-Houzel et al. 2014 — Elephant brain neuron distribution (*Frontiers Neuroanat*)
- Favila et al. 2016 — Hippocampal pattern differentiation (*Nature Comms*)
- Molitor et al. 2021 — Simultaneous DG separation + CA1 integration (*J Neuroscience*)
- Leutgeb et al. 2007 — Pattern separation in DG and CA3 (*Science*)
- Schuck et al. 2016 — OFC task-state representation (*Neuron*)
- Frank et al. 2001 — Basal ganglia gating model (*CABN*)
- Turrigiano et al. 1998 — Synaptic scaling (*Nature*)
- Patzke et al. 2015 — Cetacean hippocampus (*Brain Struct Funct*)

**SSM lineage:**
- Gu et al. 2022 — S4: Structured State Spaces for Sequence Modeling
- Gu & Dao 2023 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Dao & Gu 2024 — Mamba-2

**Sleep/consolidation in neural networks (must-cite):**
- Hinton et al. 1995 — Wake-sleep algorithm (*Science*)
- Kirkpatrick et al. 2017 — Elastic Weight Consolidation (*PNAS*)
- van de Ven et al. 2020 — Brain-inspired replay (*Nature Comms*)
- Hayes et al. 2021 — Replay survey, identifies missing biological elements (*Neural Computation*)
- Robinson et al. 2022 — NREM + REM + downscaling for continual learning (*IJCNN*)
- Tadros et al. 2022 — Sleep-like unsupervised replay (*Nature Comms*)
- Hopf et al. 2024 — WSCL: Wake-sleep consolidated learning (*IEEE TNNLS*) **[closest structural analogue]**
- Harun et al. 2023 — SIESTA: continual learning with sleep (*TMLR*)
- "Language Models Need Sleep" 2024 — Sleep as distillation + synthetic data (*OpenReview*)
- SleepGate 2026 — Sleep-inspired KV cache management for LLMs (*arXiv*) **[closest in domain]**

**Memory-augmented models:**
- Wu et al. 2022 — Memorizing Transformers (*ICLR*)
- Khandelwal et al. 2020 — kNN-LM (*ICLR*)
- Ellis et al. 2021 — DreamCoder: wake-sleep for program synthesis (*PLDI*)
