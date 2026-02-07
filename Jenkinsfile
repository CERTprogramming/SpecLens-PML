/*
=========================================================
SpecLens-PML Jenkins CI Pipeline
=========================================================

This Jenkins pipeline automates the full MLOps lifecycle:

- Reset demo state
- Rebuild datasets
- Train candidate models
- Promote champion model
- Archive ML artifacts

It represents a simplified Continuous Integration workflow
for ML-based software correctness systems.
*/

pipeline {
    agent any

    stages {

        stage("Checkout Repository") {
            steps {
                // Pull latest version of the project from Git
                echo "=== Checking out SpecLens-PML repository ==="
                checkout scm
            }
        }

        stage("Setup Python Environment") {
            steps {
                /*
                 Create an isolated virtual environment
                 to ensure reproducibility of dependencies.
                 */
                echo "=== Setting up Python virtual environment ==="

                sh """
                python3 -m venv .venv
                . .venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                """
            }
        }

        stage("Reset Pipeline State") {
            steps {
                /*
                 Reset artifacts from previous runs:
                 - feedback pool
                 - datasets
                 - candidate + champion models

                 Ensures a clean demo execution.
                 */
                echo "=== Resetting SpecLens demo state ==="

                sh """
                chmod +x reset.sh
                ./reset.sh
                """
            }
        }

        stage("Run End-to-End Demo Pipeline") {
            steps {
                /*
                 Executes the full continuous learning workflow:

                 TRAIN → TEST → Train Candidates → Promote Champion
                 → Inference on UNSEEN → Feedback Collection
                 */
                echo "=== Running full SpecLens pipeline demo ==="

                sh """
                . .venv/bin/activate
                python3 demo.py
                """
            }
        }

        stage("Verify Champion Promotion") {
            steps {
                /*
                 Governance check:
                 the pipeline must produce a promoted champion model
                 for inference and deployment.
                 */
                echo "=== Verifying champion model artifact ==="

                sh """
                if [ ! -f models/best_model.pkl ]; then
                    echo "ERROR: Champion model not found!"
                    exit 1
                fi

                echo "Champion model successfully promoted."
                """
            }
        }

        stage("Archive MLOps Artifacts") {
            steps {
                /*
                 Store generated ML artifacts for traceability:

                 - trained candidate models
                 - champion model
                 - generated datasets
                 */
                echo "=== Archiving datasets and trained models ==="

                archiveArtifacts artifacts: '''
                    models/*.pkl,
                    data/datasets_train.csv,
                    data/datasets_test.csv
                ''', fingerprint: true
            }
        }
    }

    post {

        success {
            // Pipeline completed correctly
            echo "SpecLens-PML CI pipeline completed successfully!"
        }

        failure {
            // Pipeline failed: logs help debugging reproducibility issues
            echo "Pipeline failed. Check Jenkins logs for details."
        }
    }
}

