# SpecLens — System Specification Document (SSD)

## 1. Problem Definition

SpecLens-PML is an educational data-driven system that applies
Machine Learning and MLOps principles to the domain of software
correctness.
The project introduces PML (Python Modelling Language), a
lightweight specification language inspired by JML (Java Modelling Language),
and builds an end-to-end MLOps pipeline with feedback-driven retraining.

The system analyzes Python functions annotated with PML contracts:

- `@requires` (preconditions)  
- `@ensures` (postconditions)  
- `@invariant` (class invariants)

The ML task is formulated as a binary classification problem (SAFE vs RISKY).
Operational risk levels (LOW / MEDIUM / HIGH) are derived as a post-processing layer
on top of the model prediction to support governance decisions:

- Input: structural feature vector extracted from code + contracts  
- Output: probability of being RISKY and an operational risk level (`LOW`, `MEDIUM`, `HIGH`)

SpecLens-PML provides probabilistic decision support rather than formal correctness guarantees.

---

## 2. System Context and Stakeholders

Primary stakeholders include:

- Software engineers writing annotated Python code  
- Quality Assurance (QA) and verification teams reviewing correctness risks  
- Developers experimenting with specification-driven MLOps automation  

The system operates between traditional testing and full formal verification:

- Like testing, it relies on dynamic execution
- Like specification-based approaches, it treats contracts as semantic signals  

---

## 3. Key Performance Indicators (KPIs)

The SpecLens-PML prototype focuses on the following safety-oriented and operational KPIs:

- Candidate models are compared on the held-out TEST set and only the one achieving the highest RISKY-class above a minimum threshold (configured in `config.yaml`) is promoted as the serving champion (`best_model.pkl`)
- Interactive inference remains fast on single files, whereas training and retraining costs scale with dataset growth

Given the safety-oriented domain, SpecLens prioritizes interpretable models (logistic regression, random forest) and a decision-support advisory, rather than black-box predictions.

---

## 4. Data Specification

The repository contains four pools of annotated Python examples:

- `raw_train/` (training pool)
- `raw_test/` (held-out evaluation pool)
- `raw_unseen/` (inference-only pool)
- `raw_feedback/` (collected high-risk examples)

Generated datasets:

- `datasets_train.csv`  
- `datasets_test.csv`  

These datasets are automatically produced during execution and are not tracked as static repository artifacts.

---

## 5. Label Generation

Labels are produced through dynamic execution and contract checking:

- Functions are executed on generated inputs  
- Pre / postconditions are validated  
- Contract violations detected at runtime (RISKY)
- Otherwise (SAFE)

---

## 6. Feature Extraction

Feature extraction is centralized in `pipeline/features.py` and shared across training and inference.

Features include:

- Number of parameters  
- Number of contracts (`requires`, `ensures`, `invariant`)  
- Additional structural features extracted from the code and its contracts (e.g., number of preconditions / postconditions)
- Lines of code (LOC) 

---

## 7. Requirements

Functional Requirements:

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| FR-01 | Parse PML contracts from Python code | Units parsed without errors |
| FR-02 | Extract structural features | Feature vector matches schema |
| FR-03 | Build TRAIN / TEST datasets automatically | CSV datasets generated |
| FR-04 | Train candidate models | Candidate artifacts saved |
| FR-05 | Evaluate candidates on TEST set | Metrics report produced |
| FR-06 | Promote champion model | `best_model.pkl` updated |
| FR-07 | Serve inference predictions | Risk level returned |
| FR-08 | Collect feedback from unseen inference | High-risk cases stored |

Non-Functional Requirements:

| ID | Requirement | Metric |
|----|------------|--------|
| NFR-01 | Performance | Inference latency low for individual files, while training time naturally scales with dataset size |
| NFR-02 | Data separation | TRAIN never mixed with TEST |
| NFR-03 | Configurability | Policies controlled via YAML |
| NFR-04 | Maintainability | Modular pipeline structure |
| NFR-05 | Reproducibility | Resettable via `reset.sh` |

---

## 8. Architecture Overview

The full pipeline can be executed reproducibly from scratch using the provided `reset.sh` script, which clears generated artifacts and resets the feedback loop before a new run.
The following diagram represents the implemented operational lifecycle, including the feedback loop that reinjects high-risk unseen examples into training on subsequent runs:

```mermaid
flowchart TD

    A[Python Code + PML Contracts] --> B1[Build TRAIN dataset]
    A --> B2[Build TEST dataset]

    B1 --> D[Train candidate models]
    D --> E1[logistic.pkl]
    D --> E2[forest.pkl]

    E1 --> F[Continuous Training Trigger]
    E2 --> F
    B2 --> F

    F --> G[Champion model: best_model.pkl]

    G --> H[Inference on UNSEEN pool]
    H --> I[raw_unseen/]

    H --> J[HIGH risk detected]
    J --> K[raw_feedback/]

    K --> B1
```

This diagram represents the full implemented workflow: feedback examples are collected and re-injected into TRAIN at the next run.

---

## 9. Risk Analysis

The table below summarizes four key technical and operational risks in SpecLens-PML: distribution shifts in coding and specification patterns (drift), imbalance between SAFE and RISKY examples, potential overfitting due to limited training data and the need to interpret probabilistic predictions as decision support rather than absolute guarantees, together with the corresponding mitigation strategies adopted in the pipeline:

| Risk | Impact | Mitigation |
|------|--------|------------|
| Drift in coding / spec patterns | Medium | Feedback-driven retraining |
| Class imbalance | Medium | Recall-oriented promotion |
| Overfitting on small datasets | Low | Held-out TEST evaluation |
| Misinterpretation of probabilistic outputs | Medium | Decision-support advisory |

