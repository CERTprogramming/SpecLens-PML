# SpecLens — Operational Governance & Versioning Document

## 1. Governance Scope

SpecLens-PML implements an educational governance strategy focused on:

- Candidate vs champion separation
- Metric-driven promotion
- Policy-driven governance thresholds defined in configuration (`config.yaml`)
- Controlled serving through a single deployed artifact
- Automated CI execution via containerized Jenkins pipeline
- Feedback collection for continuous retraining
- Reproducibility through reset and deterministic execution flow

The system does not include a full enterprise model registry: the promoted champion artifact is updated at each evaluation cycle.

---

## 2. Managed Artifacts

The SpecLens-PML codebase is modular and fully versioned through Git:

- Modular repository structure (`pipeline/`, `inference/`, `pml/`)  
- Versioned through Git commits and tagged releases

Training and held-out TEST datasets are generated as CSV artifacts during each pipeline execution:

  - `data/datasets_train.csv`
  - `data/datasets_test.csv`

Raw pools remain immutable:

- `raw_train/`, `raw_test/`, `raw_unseen/`

Feedback pool evolves over time:

- `raw_feedback/`

The training stage produces multiple candidate model artifacts, while governance promotes a single champion model used for operational serving:

| Type | Artifact | Role |
|------|----------|------|
| Candidate | `logistic.pkl` | Baseline model |
| Candidate | `forest.pkl` | Challenger model |
| Champion | `best_model.pkl` | Single serving model |

---

## 3. Model Lifecycle Governance

The following state diagram summarizes the governance lifecycle of SpecLens-PML models, from initial training to evaluation, champion deployment and feedback-driven retraining:

```mermaid
stateDiagram-v2
  Draft --> Trained : pipeline/train.py
  Trained --> Evaluated : ct_trigger.py
  Evaluated --> Deployed : best_model.pkl
  Deployed --> Retrained : feedback loop
```

The lifecycle enforces separation between:

- Training artifacts (candidates)  
- Production artifact (champion)  

---

## 4. Champion/Challenger Promotion Policy

Promotion is implemented in `ct_trigger.py`:

- Load candidate models  
- Evaluate on held-out TEST dataset  
- Compute Recall on the *RISKY* class  
- Promote the best candidate (`models/best_model.pkl`)

This governance rule ensures:

- Controlled deployment  
- Safety-oriented selection  
- Explicit separation between TRAIN and TEST  

---

## 5. Feedback-Driven Continuous Training Policy

Inference is performed on the UNSEEN pool:

- `data/raw_unseen/`

If a function is classified as HIGH risk, the corresponding file is copied into:

- `data/raw_feedback/`

The training pool evolves iteratively:

- The next training set is built by merging the original raw training pool with the accumulated feedback examples
- The feedback pool grows by adding unseen inputs classified as HIGH risk

The diagram below illustrates how high-risk unseen inputs are collected into the feedback pool and reinjected into the training dataset in the next continuous learning cycle:

```mermaid
flowchart LR
  U[UNSEEN pool] --> P[Predict]
  P -->|HIGH risk| F[raw_feedback/]
  F -->|next cycle| T[Expanded TRAIN dataset]
```

---

## 6. Reproducibility and Reset Controls

The full pipeline can be executed from scratch via:

```bash
./reset.sh
python3 demo.py
```

Reset removes:

- Feedback examples collected in `raw_feedback/`
- Temporary training staging directory (`data/_tmp_train/`)
- Generated TRAIN / TEST datasets (`datasets_train.csv`, `datasets_test.csv`)
- Trained candidate and champion model artifacts (`logistic.pkl`, `forest.pkl`, `best_model.pkl`)

Raw pools remain untouched, ensuring reproducible rebuilds.

---

## 7. CI/CD and Automation

SpecLens-PML integrates automation through:

- `demo.py` for end-to-end continuous training runs  
- `ct_trigger.py` for automated governance promotion  
- Jenkins integration for CI execution of the full workflow (executed inside a Docker container, ensuring that the full pipeline can be replicated in an isolated environment outside the developer’s local machine)
- Streamlit GUI (`app.py`) as an operational control interface  

---

## 8. Monitoring and Maintenance Plan

Monitoring is implemented through governance-driven signals.
Instead of relying on external observability stacks, the system reacts to:

- Performance degradation (measured through recall on the held-out TEST dataset)
- An increase of HIGH-risk unseen inputs (used to expand the feedback pool)
- And drift indicators in specification patterns, i.e., shifts in the distribution
of extracted contract features compared to the training data, by blocking champion promotion and collecting feedback examples for subsequent retraining:

| Signal | Response Action |
|--------|----------------|
| Recall drop on TEST | Block champion promotion |
| Surge of HIGH-risk unseen cases | Expand feedback pool |
| Drift suspicion in specification patterns | Trigger retraining cycle |

The feedback mechanism provides a lightweight proxy for production monitoring in an educational setting.

---

## 9. Event Log Schema (Process Mining Ready)

To support traceability, the workflow can be represented as an event log:

| timestamp | case_id | activity | artifact | outcome |
|----------|---------|----------|----------|---------|
| t1 | model_v1 | train | datasets_train.csv | success |
| t2 | model_v1 | evaluate | datasets_test.csv | recall=0.72 |
| t3 | model_v1 | promote | best_model.pkl | deployed |

This schema enables future extensions with process mining and compliance auditing.

## 10. Example Operational Use Case

A typical end-to-end interaction scenario is:

- A developer submits Python code annotated with PML contracts  
- The system performs inference using the deployed champion model  
- If the risk level is classified as HIGH, the file is copied into the feedback pool  
- The feedback pool is automatically incorporated into the next training cycle  

This lightweight scenario provides a simple form of system modeling and traceability aligned with classical Software Engineering practices.

