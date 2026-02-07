pipeline {
    agent any

    parameters {
        booleanParam(
            name: "RESET",
            defaultValue: false,
            description: "If enabled, resets datasets, models, and feedback pool (demo mode)."
        )
    }

    environment {
        PYTHONPATH = "${WORKSPACE}"
    }

    stages {

        stage("Checkout Repository") {
            steps {
                checkout scm
            }
        }

        stage("Setup Python Environment") {
            steps {
                sh """
                python3 -m venv .venv
                . .venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                """
            }
        }

        // ------------------------------------------------------------
        // Reset ONLY if requested
        // ------------------------------------------------------------
        stage("Optional Reset") {
            when {
                expression { params.RESET == true }
            }
            steps {
                echo "=== RESET ENABLED: Cleaning full pipeline state ==="

                sh """
                chmod +x reset.sh
                ./reset.sh
                """
            }
        }

        // ------------------------------------------------------------
        // Prepare TRAIN pool (keeps feedback across runs)
        // ------------------------------------------------------------
        stage("Prepare TRAIN Pool with Feedback") {
            steps {
                echo "=== Building training pool (raw_train + accumulated feedback) ==="

                sh """
                mkdir -p data/_tmp_train

                # Always start from base training set
                cp data/raw_train/*.py data/_tmp_train/ || true

                # Add feedback from previous Jenkins runs
                cp data/raw_feedback/*.py data/_tmp_train/ || true

                echo "Current TRAIN pool:"
                ls -1 data/_tmp_train || true
                """
            }
        }

        stage("Build TRAIN Dataset") {
            steps {
                sh """
                . .venv/bin/activate
                python3 pipeline/build_dataset.py \
                    data/_tmp_train \
                    data/datasets_train.csv
                """
            }
        }

        stage("Build TEST Dataset") {
            steps {
                sh """
                . .venv/bin/activate
                python3 pipeline/build_dataset.py \
                    data/raw_test \
                    data/datasets_test.csv
                """
            }
        }

        stage("Train Candidate Models") {
            steps {
                sh """
                . .venv/bin/activate
                python3 pipeline/train.py data/datasets_train.csv --model logistic
                python3 pipeline/train.py data/datasets_train.csv --model forest
                """
            }
        }

        stage("Promote Champion Model") {
            steps {
                sh """
                . .venv/bin/activate
                python3 ct_trigger.py data/datasets_test.csv
                """
            }
        }

        stage("Run Inference + Collect Feedback") {
            steps {
                sh """
                . .venv/bin/activate
                mkdir -p data/raw_feedback

                for file in data/raw_unseen/*.py; do
                    python3 inference/predict.py "$file"
                done
                """
            }
        }

        stage("Archive Artifacts") {
            steps {
                archiveArtifacts artifacts: '''
                    models/*.pkl,
                    data/datasets_train.csv,
                    data/datasets_test.csv,
                    data/raw_feedback/*.py
                ''', fingerprint: true
            }
        }
    }

    post {
        success {
            echo "Continuous Learning CI completed successfully!"
        }
        failure {
            echo "Pipeline failed."
        }
    }
}

