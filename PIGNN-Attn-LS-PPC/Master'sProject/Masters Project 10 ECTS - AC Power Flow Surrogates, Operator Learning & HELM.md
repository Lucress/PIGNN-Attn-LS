# Master's Project (10 ECTS)
## AC Power Flow Surrogates, Operator Learning & HELM

This master's project explores hybrid AC power flow surrogates that combine physics-informed operator learning with deterministic mathematical solvers such as HELM (Holomorphic Embedding Load Flow Method). Building on our current PIGNN-Attn-LS codebase and our ICASSP 2026 paper on edge-aware attention and Armijo line-search correction, the goal is to investigate whether learned operators can be combined with non-iterative or weakly iterative analytic solvers to obtain fast, robust, and physically grounded load-flow solutions. Inspired by recent ideas such as GridFM, the project aims to study how complex-analysis-based solver structure can be merged with neural models rather than treating machine learning and mathematical power-flow solvers as separate worlds.

## Project Scope

- Revisit the current physics-informed graph surrogate and reinterpret it from an operator-learning perspective.
- Investigate established operator-learning approaches for AC power flow, such as graph operator models, attention-based graph surrogates, and neural-operator-style mappings for variable-size grids.
- Study hybrid formulations in which HELM or related deterministic complex-analysis solvers provide coefficients, embeddings, constraints, warm starts, or correction targets for learned surrogates.
- Compare pure learned surrogates, hybrid learned-solver models, and classical numerical methods in terms of accuracy, residuals, runtime, and robustness.

## Prerequisites

- Strong Python skills and a solid foundation in machine learning or deep learning.
- Interest in power systems, graph learning, scientific computing, and numerical methods.
- Motivation to work on research-oriented implementation, experimentation, and analysis.
- Comfort with reading recent literature and communicating technical findings clearly.
- Familiarity with AC power flow, complex analysis, or optimization is helpful but not strictly required.

## Project Details

- Type: Master's Project
- Credits: 10 ECTS
- Topic: AC Power Flow Surrogates, Operator Learning & HELM
- Lab: Pattern Recognition Lab, Friedrich-Alexander-Universitaet Erlangen-Nuernberg (FAU)

For questions, contact changhun.kim@fau.de and siming.bayer@fau.de. The detailed project description starts below.

---

# Hybrid Operator Learning for AC Power Flow Surrogates with HELM-Based Deterministic Solvers

## Motivation

Fast and reliable AC power flow is a core requirement in planning, security assessment, uncertainty propagation, and scenario-based optimization. In many practical settings, thousands or even millions of operating points must be evaluated, which makes classical iterative solvers such as Newton-Raphson a computational bottleneck despite their strong accuracy. This has motivated a large body of work on machine-learning-based surrogates, graph neural solvers, and physics-informed models for load flow.

Our recent ICASSP 2026 work established a strong starting point in this direction. In that work, we developed a physics-informed graph neural solver with edge-aware self-attention, explicit admittance-based edge biases, and an Armijo-style backtracking line-search correction operator. The model learns a graph-structured residual-to-update operator and is trained with physics losses while remaining compatible with variable grid structures. The current research code in this project extends that line further: it supports branch-row power-system data, direct Ybus reconstruction from branch metadata, correct multi-voltage per-unit conversion, block-diagonal batching for variable-size systems, and systematic evaluation with voltage RMSE and power-mismatch residual metrics. Recent experiments in the repository already cover multiple benchmark-style PPC cases and architecture sweeps over attention heads, hidden dimensions, layers, and Armijo stabilization.

At the same time, a natural next question is whether we can move beyond purely learned iterative correction operators. Recent ideas such as GridFM suggest that deterministic, non-iterative or analytically structured load-flow formulations can be combined with neural components in a much tighter way. This is especially interesting for HELM, where complex analysis and holomorphic embedding offer a mathematically grounded alternative to Newton-style iteration. Such methods are attractive because they are deterministic, interpretable, and solver-centric, but they can still become expensive, difficult to scale, or sensitive to practical design choices. This creates an opportunity for a hybrid approach: use mathematical solvers such as HELM as structured priors or known operators, and use operator learning to model the difficult parts efficiently.

## Core Idea

The central idea of the project is to develop a hybrid framework for AC power flow in which operator learning and mathematical solvers complement each other. Instead of viewing the surrogate purely as a black-box map from injections to voltages, or viewing HELM purely as a standalone solver, we aim to study learned operator formulations that inherit structure from the physics and from deterministic solution methods.

Concretely, the student will start from our existing graph-based surrogate framework, where the network already has access to explicit admittance information through edge features, sparse graph attention, and physics-based residual evaluation. This baseline can be reinterpreted as a learned operator acting on the pair of network structure and operating point. From there, the project will investigate how HELM-style formulations can be integrated into the learning pipeline. Possible directions include learning to predict holomorphic embedding coefficients, learning corrections to truncated HELM solutions, learning a fast surrogate for selected expensive solver stages, or using HELM-derived quantities as intermediate representations that improve generalization and determinism.

The longer-term vision is a surrogate that is not only fast, but also more structured than conventional end-to-end regression. Depending on the exact design, the resulting model could be one-shot, fixed-depth, or hybrid with only a very small number of deterministic correction steps. This would connect our current known-operator, physics-informed attention framework with a more analytically grounded solver family and open a promising research direction at the intersection of machine learning, numerical analysis, and power systems.

## Scientific Objectives

The first objective is to formalize the current PIGNN-Attn-LS approach as a strong operator-learning baseline for AC power flow. The current codebase already contains the key ingredients for this: graph-structured message passing, admittance-aware edge encoding, sparse self-attention, Armijo-style line-search stabilization, explicit Ybus reconstruction, and residual-based evaluation. A first step of the project is therefore to position this model family relative to broader operator-learning literature and to define what aspects should be preserved in the next generation of hybrid surrogates.

The second objective is to investigate established operator-learning approaches for this task. Candidate directions include graph neural operators, DeepONet-style branch-trunk decompositions, attention-based graph operators, and other architectures that can learn mappings between functional or structured inputs and structured voltage solutions. A key challenge is that power grids are irregular graphs with changing sizes, heterogeneous bus types, transformer effects, and complex-valued physics, so the project should pay particular attention to architectures that remain meaningful under such variability.

The third objective is to study how HELM can be integrated into the surrogate design. This may take several forms: using HELM as a deterministic baseline whose outputs are refined by a learned correction operator; learning to predict selected series coefficients or Padé-related quantities; using HELM-inspired embeddings as latent coordinates for learning; or constructing hybrid inference pipelines in which the neural model helps accelerate or regularize a mathematically grounded solver. The exact formulation can be adapted based on feasibility and literature review, but the main goal is to go beyond a simple warm-start story and instead build a genuinely hybrid operator.

The fourth objective is to investigate deterministic or near-deterministic inference regimes. Our current model still uses unrolled correction steps, even though it is stabilized by line search. Inspired by GridFM-like ideas, an important question is whether operator learning plus HELM can move us closer to non-iterative or fixed-depth inference while maintaining physical fidelity. This is scientifically interesting because it would combine the predictability of analytic solvers with the flexibility of data-driven models.

The fifth objective is to perform a systematic empirical study across grid sizes, topologies, and operating regimes. The project should evaluate not only standard voltage errors, but also physical residuals, convergence-like failure modes, and runtime or throughput. In particular, it should examine how the proposed hybrid models behave under out-of-distribution loading, topology changes, and larger benchmark systems.

## Model Variants and Work Packages

A sensible structure is to formulate the project around several nested model families.

The first model is the current repository baseline: a physics-informed graph surrogate with edge-aware self-attention and Armijo line-search correction. This model already uses branch-row admittance information, sparse graph attention with direction-aware edge features, and residual-based evaluation metrics, making it a strong starting point.

The second model is an operator-learning variant of this baseline. Instead of emphasizing only iterative message passing, the model can be reframed as a graph operator that maps the structured pair of admittance data and power injections to voltage magnitude and angle. This stage helps establish how much can already be gained from operator-learning ideas without HELM integration.

The third model is a HELM-informed hybrid surrogate. Here, HELM is not just a baseline solver, but part of the representation or inference process. For example, the model may consume HELM-derived coefficients, approximate the mapping from system data to selected analytic coefficients, or predict residual corrections to a truncated deterministic solution.

The fourth model is a more ambitious deterministic or fixed-depth hybrid. In this setting, the goal is to explore whether the learned model can deliver a one-shot or nearly one-shot AC power flow surrogate, possibly with a final analytic correction or consistency check. This is the most directly GridFM-inspired direction and would be especially interesting if it preserves strong residual quality while reducing inference variability.

## Baselines and Comparison Design

The comparison should cover three complementary baseline families.

The first family contains classical numerical solvers, especially Newton-Raphson and HELM. Newton-Raphson remains the practical reference point for accuracy and reliability, while HELM is the key deterministic mathematical baseline for this project.

The second family contains the current learned models available in our codebase, including the MLP-based graph solver and the edge-aware self-attention model, each with and without Armijo-style stabilization. These baselines are especially important because they represent the direct continuation of our existing research line and make it possible to quantify the value of HELM integration rather than only the value of machine learning in general.

The third family contains pure operator-learning and hybrid learned-solver models developed during the project. The main comparison should clarify where structure helps most: accuracy, physical consistency, deterministic behavior, runtime, or robustness to grid and operating-point shifts.

## Data and Evaluation Scope

The project can build on two complementary data settings. First, it can use the synthetic HV/MV scenario-generation setting from our previous ICASSP work, where physically plausible operating points and reference solutions are created systematically. Second, it can use the newer branch-row pandapower/PPC-style data pipeline already implemented in the current repository. This pipeline supports variable-size systems, explicit transformer metadata, direct Ybus reconstruction, and multi-voltage per-unit handling. The current repository already contains experiments on several benchmark-style cases, including systems such as case24, case30, case39, case57, case89pegase, case118, case145, case300, and larger PEGASE-style settings, which makes it a strong basis for a broader benchmark study.

The primary evaluation criteria should include voltage-magnitude RMSE, voltage-angle RMSE, active and reactive power mismatch residuals, success rate under stressed scenarios, and runtime or throughput. Since one of the project goals is to combine learning with deterministic solver structure, it is also important to assess how predictable and stable the inference process is across repeated runs, changing operating points, and system sizes.

## Expected Methodological Contribution

The main methodological contribution is not simply another AC power flow surrogate, but a hybrid research framework that connects operator learning with mathematically grounded deterministic solvers. The novelty lies in combining the current strengths of our codebase and prior paper, namely physics-informed graph learning, known-operator edge encoding, and stabilized correction steps, with HELM-inspired analytic structure and recent non-iterative solver ideas.

If successful, the project would help clarify when a learned operator should replace a solver, when it should assist a solver, and how the two can be fused in a principled way. This would be valuable not only for fast load-flow approximation, but also for broader questions in scientific machine learning where deterministic operators and neural surrogates need to coexist rather than compete.
