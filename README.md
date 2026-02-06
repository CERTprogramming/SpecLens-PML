# SpecLens-PML

**SpecLens-PML** is an educational data-driven system that applies
Machine Learning and MLOps principles to the domain of software
correctness.

The project introduces **PML (Python Modelling Language)**, a
lightweight specification language inspired by JML, and builds an
end-to-end pipeline that:

- ingests Python code annotated with PML,
- treats code and specifications as data,
- generates a dataset through dynamic execution,
- trains a machine learning model,
- predicts the likelihood of contract violations in new code.

Unlike formal verification tools, SpecLens-PML does **not** aim to prove
correctness.  
Due to Python's dynamic nature (lack of strong static typing,
runtime-dependent semantics), building an external prover is
impractical.  
SpecLens-PML embraces this reality and provides *probabilistic
guidance*, helping developers identify risky functions before runtime
failures occur.

This repository demonstrates a **complete MLOps lifecycle**: data
generation, training, versioning, inference, monitoring, continuous
retraining, and user-facing serving.

------------------------------------------------------------------------

## Project Structure

```
spec-lens-pml/
├── app.py                  # Streamlit web interface
├── config.yaml             # Centralized system configuration
├── ct_trigger.py           # Continuous Training & promotion engine
├── data/
│   ├── raw/                # Python files annotated with PML
│   └── datasets_v1.csv     # Generated dataset (versioned)
├── pml/
│   └── parser.py           # AST + PML parser
├── pipeline/
│   ├── build_dataset.py    # Data generation and labeling
│   └── train.py            # Model training (version-aware)
├── inference/
│   └── predict.py          # Inference using the active model
├── models/
│   ├── model_vN.pkl        # Versioned model artifacts
│   └── active_model.txt    # Pointer to the active model
├── demo.py                 # End-to-end CLI demo script
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## PML Syntax Example

```python
def div(a, b):
    # @requires b != 0
    # @ensures result * b == a
    return a // b
```

```python
class Account:
    # @invariant self.balance >= 0

    def withdraw(self, amount):
        # @requires amount > 0
        # @ensures self.balance >= 0
        self.balance -= amount
```

Supported annotations:

- `@requires <expr>` -- preconditions  
- `@ensures <expr>` -- postconditions  
- `@invariant <expr>` -- class invariants

Expressions are a lightweight subset of Python boolean expressions.

------------------------------------------------------------------------

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```txt
pandas
scikit-learn
joblib
PyYAML
streamlit
```

Initialize the active model (once):

```bash
echo "models/model_v1.pkl" > models/active_model.txt
```

------------------------------------------------------------------------

## End-to-End Demo (CLI)

The entire MLOps pipeline can be executed with a single command:

```bash
python3 demo.py
```

This performs:

1. **Dataset Generation**

   ```bash
   python pipeline/build_dataset.py data/raw data/datasets_v1.csv
   ```

   - Parses Python files with PML.
   - Executes functions with generated inputs.
   - Evaluates specifications.
   - Assigns labels based on observed violations.
   - Produces a versioned dataset.

2. **Model Training**

   ```bash
   python pipeline/train.py data/datasets_v1.csv
   ```

   - Reads hyperparameters from `config.yaml`.
   - Trains a baseline ML classifier.
   - Prints evaluation metrics.
   - Saves a *new versioned model* (`model_vN.pkl`).

3. **Inference on New Code**

   ```bash
   python inference/predict.py data/raw/example.py
   ```

   - Loads the model pointed by `models/active_model.txt`.
   - Parses new code.
   - Reconstructs the same feature vector used in training.
   - Outputs:
     - a numeric risk score in [0,1]
     - an operational risk level: `LOW`, `MEDIUM`, `HIGH`.

The demo automatically runs all three steps on all files in `data/raw/`.

------------------------------------------------------------------------

## Web Interface (Streamlit)

SpecLens-PML also provides a lightweight **web GUI** implemented with
Streamlit. The GUI does not replace the MLOps pipeline: it is a thin
presentation layer on top of the existing backend components.

Start the web application with:

```bash
streamlit run app.py
```

The interface exposes the full system to non-technical users:

- **Run full pipeline**  
  Executes `demo.py` (dataset generation + training + inference).

- **Trigger Continuous Training**  
  Executes `ct_trigger.py`, potentially retraining and promoting a new
  model.

- **Active model display**  
  Shows the model currently in production (from `active_model.txt`).

- **Code analysis**  
  Upload a `.py` file annotated with PML and obtain:
  - function-level analysis,
  - risk scores,
  - operational levels (`LOW`, `MEDIUM`, `HIGH`).

The Streamlit application reuses the same backend scripts:

- `demo.py`
- `ct_trigger.py`
- `predict.py`

No MLOps logic is duplicated or altered. The GUI only changes *how the
system is operated*, not *how it behaves*.

------------------------------------------------------------------------

## MLOps Lifecycle

SpecLens-PML implements a complete MLOps workflow:

1. **Data Pipeline**
   - Code + specifications are treated as data.
   - Datasets are generated and versioned (`datasets_vN.csv`).

2. **ML Kernel**
   - Training is fully reproducible.
   - Hyperparameters and thresholds are centralized in `config.yaml`.

3. **Model Versioning**
   - Each training run produces a new artifact:
     ```
     models/model_v1.pkl
     models/model_v2.pkl
     ...
     ```
   - Older models are preserved to enable rollback.

4. **Operational Semantics**
   - Predictions are mapped to decision levels:
     - `LOW`    – acceptable risk
     - `MEDIUM` – warning
     - `HIGH`   – critical
   - The system provides *decision support*, not proofs.

5. **Continuous Training**

```bash
python3 ct_trigger.py
```

The Continuous Training component:

- monitors model performance,
- compares metrics with operational thresholds,
- retrains the model when required,
- evaluates the new model against the active one,
- promotes the new model only if it improves safety-oriented metrics.

This design enables:

- reproducibility,
- traceability,
- rollback,
- adaptation to data drift,
- controlled deployment.

------------------------------------------------------------------------

## Training vs Serving

SpecLens-PML explicitly separates **model training** from **model
serving**.

- The training pipeline may generate many model versions:
  ```
  models/model_v1.pkl
  models/model_v2.pkl
  models/model_v13.pkl
  ...
  ```

- The inference layer does **not** automatically use the latest trained
  model.

Instead, the model currently in production is defined by:

```
models/active_model.txt
```

This file contains the path of the *active* model, for example:

```
models/model_v1.pkl
```

`predict.py` always loads the model specified in `active_model.txt`.
This ensures that:

- training does not implicitly change system behavior,
- new models are never deployed by accident,
- rollback is immediate (just update one file),
- governance policies can be enforced.

### Model Promotion

`ct_trigger.py` closes the loop:

```bash
python3 ct_trigger.py
```

It performs the following steps:

1. Triggers a new training run.
2. Identifies the newly produced model.
3. Evaluates both:
   - the current active model,
   - the newly trained model.
4. Compares a safety-oriented metric (Recall on the RISKY class).
5. Promotes the new model **only if it improves the metric**.

If the new model is better, `active_model.txt` is automatically updated.
Otherwise, the new model is kept for traceability but not deployed.

This implements a simplified but complete MLOps governance loop with:

- controlled deployment,
- automatic promotion,
- human-in-the-loop readiness,
- full traceability and rollback.

Training and serving are therefore *decoupled by design*, reflecting
real-world MLOps practice in safety-oriented systems.

------------------------------------------------------------------------

## Educational Scope

SpecLens-PML is designed as an educational MLOps system:

- datasets are generated automatically from code,
- labels come from dynamic execution,
- models are versioned and reproducible,
- the entire lifecycle is observable and repeatable.

The quality of predictions depends on data availability:  
as more annotated code is added to `data/raw/`, the system becomes more
informative.

The focus of the project is on **architecture, reproducibility, and
lifecycle management**, not on achieving state-of-the-art model
performance.
