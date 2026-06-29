"""
Training Script
Run this to train both CNN and ResNet models on your dataset.

Usage:
    python backend/ml/train.py --data_dir data/raw --epochs 50 --batch_size 32

Dataset expected structure:
    data/raw/
        Normal/       ← images of normal lungs
        Pneumonia/    ← images of pneumonia
        Tuberculosis/ ← images of TB
        COVID-19/     ← images of COVID-19
        Lung Cancer/  ← images of lung cancer
        COPD/         ← images of COPD
        Pleural Effusion/ ← images of pleural effusion

Recommended free datasets (download separately):
    - NIH ChestX-ray14  : https://nihcc.app.box.com/v/ChestXray-NIHCC
    - CheXpert           : https://stanfordmlgroup.github.io/competitions/chexpert/
    - RSNA Pneumonia     : https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    - COVID-19 Kaggle    : https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
"""

import argparse
import sys
import logging
from pathlib import Path

# Allow imports from parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.preprocessing import DatasetPreprocessor
from ml.models import train_and_select
from database.connection import init_db, AsyncSessionLocal, ModelMetrics
import asyncio
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


async def save_metrics_to_db(results: dict):
    """Persist training metrics into the database."""
    async with AsyncSessionLocal() as session:
        for model_name in ["cnn", "resnet"]:
            m = results[model_name]
            metric = ModelMetrics(
                model_name=model_name.upper(),
                version="1.0.0",
                accuracy=m["accuracy"],
                precision=m["precision"],
                recall=m["recall"],
                f1_score=m["f1_score"],
                auc_roc=m["auc_roc"],
                training_loss=m.get("training_loss"),
                validation_loss=m.get("validation_loss"),
                confusion_matrix=m.get("confusion_matrix"),
                class_names=m.get("class_names"),
                is_active=(model_name.upper() == results["selected_model"])
            )
            session.add(metric)
        await session.commit()
    logger.info("Metrics saved to database.")


def main():
    parser = argparse.ArgumentParser(description="Train Lung Disease Detection Models")
    parser.add_argument("--data_dir",   type=str, default="data/raw",  help="Path to raw dataset directory")
    parser.add_argument("--epochs",     type=int, default=50,           help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,           help="Training batch size")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please download a dataset and organize it into class subdirectories.")
        sys.exit(1)

    # Preprocessing
    preprocessor = DatasetPreprocessor(
        data_dir=str(data_dir),
        test_size=0.15,
        val_size=0.15
    )
    data = preprocessor.run(batch_size=args.batch_size)

    # Train both models & select best
    results = train_and_select(
        data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Save metrics to DB
    asyncio.run(init_db())
    asyncio.run(save_metrics_to_db(results))

    logger.info("\n" + "="*55)
    logger.info("  TRAINING SUMMARY")
    logger.info("="*55)
    logger.info(f"  CNN    — Accuracy: {results['cnn']['accuracy']*100:.2f}%  F1: {results['cnn']['f1_score']*100:.2f}%")
    logger.info(f"  ResNet — Accuracy: {results['resnet']['accuracy']*100:.2f}%  F1: {results['resnet']['f1_score']*100:.2f}%")
    logger.info(f"  ✅ Selected model : {results['selected_model']}")
    logger.info("="*55)
    logger.info("  Output files:")
    logger.info("    models/cnn_model.h5")
    logger.info("    models/resnet_model.h5")
    logger.info("    models/training_results.json")
    logger.info("    docs/training_curves.png")
    logger.info("    docs/confusion_matrix_CNN.png")
    logger.info("    docs/confusion_matrix_ResNet.png")


if __name__ == "__main__":
    main()
