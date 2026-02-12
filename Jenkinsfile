/*
=========================================================
SpecLens-PML Jenkins CI Pipeline
=========================================================

This Jenkins pipeline automates the full Continuous Learning loop:

   train → test → promote → unseen → feedback → retrain

Key idea:

- Feedback is accumulated across runs
- Reset is optional (manual)

This is a simplified but realistic MLOps workflow.
*/

pipeline {

    /*
    ---------------------------------------------------------
    Run on any available Jenkins agent (container node)
    ---------------------------------------------------------
    */

    agent any


    /*
    ---------------------------------------------------------
    Optional parameter:

    - RESET_DEMO=true will wipe all artifacts and feedback
    - Default is false, so the model improves across builds
    ---------------------------------------------------------
    */

    parameters {
        booleanParam(
            name: "RESET_DEMO",
            defaultValue: false,
            description: "If true, resets datasets/models/feedback before training"
        )
    }


    stages {

        /*
        =====================================================
        1. Checkout Repository
        =====================================================

        Jenkins pulls the latest version of SpecLens-PML
        from GitHub, including the Jenkinsfile itself.
        */

        stage("Checkout Repository") {
            steps {
                echo "=== Checking out SpecLens-PML repository ==="
                checkout scm
            }
        }


        /*
        =====================================================
        2. Setup Python Environment
        =====================================================

        Creates a reproducible Python virtual environment
        inside the Jenkins workspace.

        This ensures dependencies are isolated and consistent.
        */

        stage("Setup Python Environment") {
            steps {
                echo "=== Setting up Python virtual environment ==="

                sh """
                python3 -m venv .venv
                . .venv/bin/activate
                pip install --upgrade pip

                # Install SpecLens-PML as an editable package
                pip install -e .
                """
            }
        }


        /*
        =====================================================
        3. Optional Reset (Manual Governance Step)
        =====================================================

        Resetting removes:

        - accumulated feedback examples
        - generated datasets
        - trained candidate models
        - promoted champion model

        IMPORTANT:
        This stage runs ONLY if RESET_DEMO=true.
        Otherwise, feedback is preserved across builds.
        */

        stage("Optional Reset") {
            when {
                expression { params.RESET_DEMO == true }
            }

            steps {
                echo "=== Resetting demo state ==="

                sh """
                chmod +x reset.sh
                ./reset.sh
                """
            }
        }


        /*
        =====================================================
        4. Prepare TRAIN Pool with Feedback
        =====================================================

        Continuous Learning principle:

        Training data = raw_train + accumulated raw_feedback

        Feedback comes from previous inference runs,
        so the model improves over time.
        */

        stage("Prepare TRAIN Pool with Feedback") {
            steps {
                echo "=== Building training pool (raw_train + accumulated feedback) ==="

                sh """
                mkdir -p data/_tmp_train

                # Base training examples
                cp data/raw_train/*.py data/_tmp_train/ || true

                # Feedback examples collected from previous runs
                cp data/raw_feedback/*.py data/_tmp_train/ || true

                echo "Current TRAIN pool:"
                ls -1 data/_tmp_train
                """
            }
        }


        /*
        =====================================================
        5. Build TRAIN Dataset
        =====================================================

        Converts contract-annotated Python functions into
        numeric ML features + risk labels.

        Output:

        - data/datasets_train.csv
        */

        stage("Build TRAIN Dataset") {
            steps {
                sh """
                . .venv/bin/activate

                # Ensure imports work correctly inside Jenkins
                export PYTHONPATH=\$WORKSPACE

                python3 pipeline/build_dataset.py \
                    data/_tmp_train \
                    data/datasets_train.csv
                """
            }
        }


        /*
        =====================================================
        6. Build TEST Dataset (Held-Out Governance Split)
        =====================================================

        TEST is never used for training.

        It is only used to evaluate candidate models
        before promotion.

        Output:
        - data/datasets_test.csv
        */

        stage("Build TEST Dataset") {
            steps {
                sh """
                . .venv/bin/activate
                export PYTHONPATH=\$WORKSPACE

                python3 pipeline/build_dataset.py \
                    data/raw_test \
                    data/datasets_test.csv
                """
            }
        }


        /*
        =====================================================
        7. Train Candidate Models
        =====================================================

        Two candidate families are trained:

        - Logistic Regression (baseline)
        - Random Forest (challenger)

        Output artifacts:
        - models/logistic.pkl
        - models/forest.pkl
        */

        stage("Train Candidate Models") {
            steps {
                sh """
                . .venv/bin/activate
                export PYTHONPATH=\$WORKSPACE

                python3 pipeline/train.py data/datasets_train.csv --model logistic
                python3 pipeline/train.py data/datasets_train.csv --model forest
                """
            }
        }


        /*
        =====================================================
        8. Promote Champion Model (Continuous Training Trigger)
        =====================================================

        Governance decision step:

        - Evaluate candidates on TEST
        - Select best model by recall on the RISKY class
        - Promote champion for serving

        Output:
        - models/best_model.pkl
        */

        stage("Promote Champion Model") {
            steps {
                sh """
                . .venv/bin/activate
                export PYTHONPATH=\$WORKSPACE

                python3 ct_trigger.py data/datasets_test.csv
                """
            }
        }


        /*
        =====================================================
        9. Run Inference + Collect Feedback
        =====================================================

        The champion model is applied to UNSEEN examples.

        If a HIGH-risk function is detected,
        the example is copied into:

        - data/raw_feedback/

        This is the feedback loop that enables
        continuous improvement across builds.
        */

        stage("Run Inference + Collect Feedback") {
            steps {
                echo "=== Running inference on unseen examples ==="

                sh """
                . .venv/bin/activate
                export PYTHONPATH=\$WORKSPACE

                for f in data/raw_unseen/*.py; do
                    echo "Analyzing \$f"
                    python3 inference/predict.py "\$f"
                done
                """
            }
        }


        /*
        =====================================================
        10. Archive Artifacts (Traceability)
        =====================================================

        Stores datasets and trained models as Jenkins artifacts.

        This provides:

        - reproducibility
        - audit trail
        - experiment tracking
        */

        stage("Archive Artifacts") {
            steps {
                echo "=== Archiving ML artifacts ==="

                archiveArtifacts artifacts: '''
                    models/*.pkl,
                    data/datasets_train.csv,
                    data/datasets_test.csv
                ''', fingerprint: true
            }
        }
    }


    /*
    =========================================================
    Post Actions
    =========================================================
    */

    post {

        success {
            echo "SpecLens-PML Continuous Learning pipeline completed successfully!"
        }

        failure {
            echo "Pipeline failed. Check Jenkins logs for debugging."
        }
    }
}

