"""
Data Preprocessing Pipeline for Lung Images
Handles: loading, cleaning, augmentation, normalization, train/val/test split
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)         # Standard input size for CNN & ResNet
MEAN = np.array([0.485, 0.456, 0.406])   # ImageNet normalization mean
STD  = np.array([0.229, 0.224, 0.225])   # ImageNet normalization std

DISEASE_CLASSES = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
    "Lung Cancer",
    "COPD",
    "Pleural Effusion"
]

PRECAUTIONS_MAP = {
    "Normal": [
        "Continue regular health check-ups every 6–12 months.",
        "Maintain a smoke-free lifestyle.",
        "Exercise regularly to keep lungs healthy.",
        "Ensure proper ventilation in living/working spaces."
    ],
    "Pneumonia": [
        "Seek immediate medical attention for antibiotic prescription.",
        "Rest and stay well hydrated (2–3 litres of fluids per day).",
        "Avoid exposure to cold air and crowds.",
        "Monitor oxygen saturation; seek emergency care if below 94%.",
        "Complete the full antibiotic course even if symptoms improve."
    ],
    "Tuberculosis": [
        "Isolate yourself to avoid spreading infection.",
        "Begin DOTS (Directly Observed Treatment Short-course) immediately.",
        "Wear N95 mask around others.",
        "Notify close contacts for screening.",
        "Complete the 6–9 month treatment course without interruption."
    ],
    "COVID-19": [
        "Self-isolate for at least 10 days from symptom onset.",
        "Monitor oxygen levels with a pulse oximeter.",
        "Seek hospital care if SpO2 drops below 94%.",
        "Stay hydrated and rest adequately.",
        "Follow local health authority guidelines."
    ],
    "Lung Cancer": [
        "Consult an oncologist immediately for staging and treatment plan.",
        "Quit smoking completely — this is critical.",
        "Discuss biopsy and PET scan with your physician.",
        "Explore clinical trials if eligible.",
        "Seek psychological and palliative care support."
    ],
    "COPD": [
        "Use prescribed bronchodilators and inhalers as directed.",
        "Quit smoking — the single most impactful intervention.",
        "Enrol in a pulmonary rehabilitation programme.",
        "Get annual flu and pneumococcal vaccinations.",
        "Avoid air pollutants, dust, and chemical fumes."
    ],
    "Pleural Effusion": [
        "Consult a pulmonologist to identify the underlying cause.",
        "Thoracentesis (fluid drainage) may be required.",
        "Limit strenuous physical activity until reviewed.",
        "Monitor breathing difficulty and report worsening immediately.",
        "Address underlying causes such as heart failure or infection."
    ]
}


# ─── Single Image Preprocessing ───────────────────────────────────────────────

class ImagePreprocessor:
    """Preprocesses a single lung image for model inference."""

    def __init__(self, target_size: Tuple[int, int] = IMAGE_SIZE):
        self.target_size = target_size

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from path, return None on failure."""
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return img

    def load_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image directly from raw bytes (for API uploads)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes.")
        return img

    def clean_image(self, img: np.ndarray) -> np.ndarray:
        """
        Data Cleaning Steps:
        1. Convert BGR → RGB
        2. Remove noise with Gaussian blur
        3. Apply CLAHE for contrast enhancement (important for X-rays)
        4. Handle grayscale X-rays by converting to 3-channel
        """
        # Step 1: Handle grayscale images (most X-rays are grayscale)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Step 2: BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Step 3: Denoise
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Step 4: CLAHE on luminance channel for X-ray enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return img

    def resize_and_normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalization Steps:
        1. Resize to target size
        2. Normalize to [0,1]
        3. Apply ImageNet mean/std normalization
        4. Add batch dimension
        """
        # Resize
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # ImageNet normalization
        img = (img - MEAN) / STD

        # Add batch dimension: (H, W, C) → (1, H, W, C)
        img = np.expand_dims(img, axis=0)

        return img

    def preprocess(self, image_input) -> np.ndarray:
        """Full preprocessing pipeline: load → clean → normalize."""
        if isinstance(image_input, bytes):
            img = self.load_from_bytes(image_input)
        elif isinstance(image_input, str):
            img = self.load_image(image_input)
        else:
            img = image_input  # Already numpy array

        img = self.clean_image(img)
        img = self.resize_and_normalize(img)
        return img


# ─── Dataset Preprocessing (for Training) ─────────────────────────────────────

class DatasetPreprocessor:
    """
    Full dataset preprocessing pipeline for model training.
    Expects dataset directory structure:
        data/raw/
            Normal/
                img1.jpg, img2.jpg, ...
            Pneumonia/
                img1.jpg, ...
            ...
    """

    def __init__(
        self,
        data_dir: str,
        target_size: Tuple[int, int] = IMAGE_SIZE,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.preprocessor = ImagePreprocessor(target_size)
        self.label_encoder = LabelEncoder()

    # ── Step 1: Scan Dataset ──────────────────────────────────────────────────
    def scan_dataset(self) -> pd.DataFrame:
        """
        Scan dataset directory and build a DataFrame of image paths + labels.
        Also performs data quality checks.
        """
        records = []
        supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in supported_ext:
                    continue
                records.append({
                    "image_path": str(img_path),
                    "label": class_name,
                    "filename": img_path.name
                })

        df = pd.DataFrame(records)
        logger.info(f"Dataset scanned: {len(df)} images across {df['label'].nunique()} classes")
        logger.info(f"Class distribution:\n{df['label'].value_counts()}")
        return df

    # ── Step 2: Data Cleaning ─────────────────────────────────────────────────
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data cleaning steps:
        1. Remove duplicate file paths
        2. Remove corrupted / unreadable images
        3. Remove images that are too small (< 50x50)
        4. Log cleaning statistics
        """
        original_count = len(df)

        # Remove duplicates
        df = df.drop_duplicates(subset=["image_path"])
        logger.info(f"After dedup: {len(df)} (removed {original_count - len(df)})")

        # Validate images
        valid_indices = []
        for idx, row in df.iterrows():
            try:
                img = cv2.imread(row["image_path"])
                if img is None:
                    logger.warning(f"Corrupted: {row['image_path']}")
                    continue
                h, w = img.shape[:2]
                if h < 50 or w < 50:
                    logger.warning(f"Too small ({h}x{w}): {row['image_path']}")
                    continue
                valid_indices.append(idx)
            except Exception as e:
                logger.error(f"Error reading {row['image_path']}: {e}")

        df = df.loc[valid_indices].reset_index(drop=True)
        logger.info(f"After cleaning: {len(df)} valid images (removed {original_count - len(df)} total)")
        return df

    # ── Step 3: Split Dataset ─────────────────────────────────────────────────
    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into train / validation / test sets (stratified by class)."""
        train_val, test = train_test_split(
            df, test_size=self.test_size,
            stratify=df["label"], random_state=self.random_state
        )
        adjusted_val = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val, test_size=adjusted_val,
            stratify=train_val["label"], random_state=self.random_state
        )
        logger.info(f"Split — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test

    # ── Step 4: Encode Labels ─────────────────────────────────────────────────
    def encode_labels(self, df: pd.DataFrame) -> np.ndarray:
        return self.label_encoder.fit_transform(df["label"].values)

    # ── Step 5: Load & Preprocess Images ─────────────────────────────────────
    def load_images(self, df: pd.DataFrame, augment: bool = False) -> np.ndarray:
        """Load and preprocess all images in a DataFrame."""
        images = []
        for _, row in df.iterrows():
            img = cv2.imread(row["image_path"])
            if img is not None:
                img = self.preprocessor.clean_image(img)
                img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
                img_norm = img_resized.astype(np.float32) / 255.0
                img_norm = (img_norm - MEAN) / STD
                images.append(img_norm)

                if augment:
                    images.extend(self._augment(img_norm))

        return np.array(images)

    # ── Step 6: Augmentation ──────────────────────────────────────────────────
    def _augment(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Data augmentation to increase dataset diversity:
        - Horizontal flip
        - Rotation ±15°
        - Brightness/contrast jitter
        """
        augmented = []

        # Flip
        augmented.append(np.fliplr(img))

        # Rotation
        h, w = img.shape[:2]
        for angle in [-15, 15]:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            augmented.append(rotated)

        # Brightness jitter
        bright = np.clip(img * 1.2, 0, 1)
        augmented.append(bright)

        return augmented

    # ── Full Pipeline ─────────────────────────────────────────────────────────
    def run(self) -> Dict:
        """Execute the full preprocessing pipeline."""
        logger.info("=== Starting Data Preprocessing Pipeline ===")

        df = self.scan_dataset()
        df = self.clean_dataset(df)

        train_df, val_df, test_df = self.split_dataset(df)

        logger.info("Loading and preprocessing training images (with augmentation)...")
        X_train = self.load_images(train_df, augment=True)
        X_val   = self.load_images(val_df,   augment=False)
        X_test  = self.load_images(test_df,  augment=False)

        self.label_encoder.fit(df["label"])
        y_train = self.label_encoder.transform(train_df["label"].values)
        y_val   = self.label_encoder.transform(val_df["label"].values)
        y_test  = self.label_encoder.transform(test_df["label"].values)

        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_val shape:   {X_val.shape}")
        logger.info(f"X_test shape:  {X_test.shape}")
        logger.info("=== Preprocessing Complete ===")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
            "class_names": list(self.label_encoder.classes_),
            "label_encoder": self.label_encoder,
            "train_df": train_df, "val_df": val_df, "test_df": test_df
        }
