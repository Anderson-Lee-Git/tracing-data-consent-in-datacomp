import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from torchvision import transforms
from pathlib import Path
import argparse


# Dataset class
class WatermarkDataset(Dataset):
    def __init__(self, data_dir, processor):
        self.data_dir = data_dir
        self.processor = processor

        # Get all image paths and labels
        self.watermark_dir = os.path.join(data_dir, "watermark")
        self.clean_dir = os.path.join(data_dir, "clean")

        self.watermark_images = [
            (os.path.join(self.watermark_dir, f), 1)
            for f in os.listdir(self.watermark_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        self.clean_images = [
            (os.path.join(self.clean_dir, f), 0)
            for f in os.listdir(self.clean_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]

        self.samples = self.watermark_images + self.clean_images

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    0, translate=(0.1, 0.1), scale=(0.5, 1.5), shear=10
                ),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Read image in BGR format using cv2
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image at {img_path}")

        # Convert from numpy array to PIL Image while keeping BGR order
        image = Image.fromarray(image)

        processed = self.processor(images=image, return_tensors="pt")
        # augment image (B x 3 x 256 x 256) with a set of transformations
        # processed["pixel_values"] = self.transform(processed["pixel_values"])
        return {
            "pixel_values": processed["pixel_values"].squeeze(),
            "labels": torch.tensor(label),
        }


def create_stratified_split(dataset):
    # Extract labels
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    # First split into train and temp (val+test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=42
    )

    # Split temp into val and test
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # log number of clean, watermark samples in each split
    # formatted in (clean, watermark)
    # Count labels for each split
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]

    print(
        f"Number of clean samples in train: {train_labels.count(0)}, Number of watermark samples in train: {train_labels.count(1)}"
    )
    print(
        f"Number of clean samples in val: {val_labels.count(0)}, Number of watermark samples in val: {val_labels.count(1)}"
    )
    print(
        f"Number of clean samples in test: {test_labels.count(0)}, Number of watermark samples in test: {test_labels.count(1)}"
    )

    return train_idx, val_idx, test_idx


def calculate_metrics(predictions, labels):
    # Convert tensors to numpy arrays for easier calculation
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    # Calculate metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = np.mean(predictions == labels)

    return accuracy, precision, recall


def train(args):
    # Model and processor initialization
    processor = MobileViTImageProcessor.from_pretrained(
        "apple/mobilevitv2-1.0-imagenet1k-256",
        cache_dir=os.getenv("TRANSFORMERS_CACHE"),
    )
    model = MobileViTV2ForImageClassification.from_pretrained(
        (
            "apple/mobilevitv2-1.0-imagenet1k-256"
            if not args.pretrained_checkpoint
            else args.pretrained_checkpoint
        ),
        num_labels=2,
        ignore_mismatched_sizes=True,
        cache_dir=os.getenv("TRANSFORMERS_CACHE"),
    )

    # Add class weights (weight factor for label 1)
    weight_factor = (
        args.pos_class_weight
    )  # Adjust this value to control the upweighting
    class_weights = torch.tensor(
        [1.0, weight_factor]
    )  # [weight_label_0, weight_label_1]

    # Dataset preparation
    data_dir = args.training_data_dir
    full_dataset = WatermarkDataset(data_dir, processor)

    # Create stratified splits
    train_indices, val_indices, _ = create_stratified_split(full_dataset)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    class_weights = class_weights.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Best validation checkpoint
    best_val_f1_score = 0

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        train_predictions = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            # Calculate weighted loss manually
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            train_predictions.extend(predictions)
            train_labels.extend(labels)

        # Convert lists to tensors for metric calculation
        train_predictions = torch.stack(train_predictions)
        train_labels = torch.stack(train_labels)
        train_accuracy, train_precision, train_recall = calculate_metrics(
            train_predictions, train_labels
        )

        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values)
                # Calculate validation loss manually like in training
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits, labels)
                val_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                val_predictions.extend(predictions)
                val_labels.extend(labels)

        # Convert lists to tensors for metric calculation
        val_predictions = torch.stack(val_predictions)
        val_labels = torch.stack(val_labels)
        val_accuracy, val_precision, val_recall = calculate_metrics(
            val_predictions, val_labels
        )

        # Calculate F1 score
        val_f1_score = (
            2 * (val_precision * val_recall) / (val_precision + val_recall)
            if (val_precision + val_recall) > 0
            else 0
        )

        # Update best validation checkpoint if F1 score is higher
        if val_f1_score > best_val_f1_score:
            best_val_f1_score = val_f1_score
            # Save the model
            model.save_pretrained(
                args.save_dir,
                cache_dir=os.getenv("TRANSFORMERS_CACHE"),
            )

        # Print metrics
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(
            f"Train Metrics - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}"
        )
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(
            f"Val Metrics - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
        )


def evaluate(args):
    # Test evaluation on best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileViTV2ForImageClassification.from_pretrained(
        args.save_dir,
        cache_dir=os.getenv("TRANSFORMERS_CACHE"),
    ).to(device)
    model.eval()
    processor = MobileViTImageProcessor.from_pretrained(
        "apple/mobilevitv2-1.0-imagenet1k-256",
        cache_dir=os.getenv("TRANSFORMERS_CACHE"),
    )

    # Dataset preparation
    data_dir = args.training_data_dir
    full_dataset = WatermarkDataset(data_dir, processor)
    _, _, test_indices = create_stratified_split(full_dataset)

    test_dataset = Subset(full_dataset, test_indices)

    # Load wm-nowm evaluation dataset
    eval_dir = (
        Path(__file__).parent.parent / "yolov8-scripts/generate_input/wm-nowm/valid"
    )
    eval_dataset = WatermarkDataset(eval_dir, processor)

    # Load datacomp evaluation dataset
    datacomp_eval_dir = (
        Path(__file__).parent.parent
        / "yolov8-scripts/generate_input/datacomp-validation"
    )
    datacomp_eval_dataset = WatermarkDataset(datacomp_eval_dir, processor)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    datacomp_eval_loader = DataLoader(datacomp_eval_dataset, batch_size=args.batch_size)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            predictions = torch.argmax(outputs.logits, dim=1)
            test_predictions.extend(predictions)
            test_labels.extend(labels)

    # Convert lists to tensors for metric calculation
    test_predictions = torch.stack(test_predictions)
    test_labels = torch.stack(test_labels)
    test_accuracy, test_precision, test_recall = calculate_metrics(
        test_predictions, test_labels
    )

    print(
        f"Test Metrics - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}"
    )

    # wm-nowm evaluation
    wm_nowm_predictions = []
    wm_nowm_labels = []
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            predictions = torch.argmax(outputs.logits, dim=1)
            wm_nowm_predictions.extend(predictions)
            wm_nowm_labels.extend(labels)

    # Convert lists to tensors for metric calculation
    wm_nowm_predictions = torch.stack(wm_nowm_predictions)
    wm_nowm_labels = torch.stack(wm_nowm_labels)
    wm_nowm_accuracy, wm_nowm_precision, wm_nowm_recall = calculate_metrics(
        wm_nowm_predictions, wm_nowm_labels
    )
    print(
        f"wm-nowm validation set Evaluation Metrics - Accuracy: {wm_nowm_accuracy:.4f}, Precision: {wm_nowm_precision:.4f}, Recall: {wm_nowm_recall:.4f}"
    )

    # Datacomp evaluation
    datacomp_eval_predictions = []
    datacomp_eval_labels = []

    with torch.no_grad():
        for batch in datacomp_eval_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            predictions = torch.argmax(outputs.logits, dim=1)
            datacomp_eval_predictions.extend(predictions)
            datacomp_eval_labels.extend(labels)

    # Convert lists to tensors for metric calculation
    datacomp_eval_predictions = torch.stack(datacomp_eval_predictions)
    datacomp_eval_labels = torch.stack(datacomp_eval_labels)
    datacomp_eval_accuracy, datacomp_eval_precision, datacomp_eval_recall = (
        calculate_metrics(datacomp_eval_predictions, datacomp_eval_labels)
    )

    print(
        f"Datacomp Evaluation Metrics - Accuracy: {datacomp_eval_accuracy:.4f}, Precision: {datacomp_eval_precision:.4f}, Recall: {datacomp_eval_recall:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train watermark detection model")
    parser.add_argument(
        "--pretrained-checkpoint", type=str, default=None, help="pretrained checkpoint"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size for training"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="number of epochs to train"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="weight decay for optimizer"
    )
    parser.add_argument(
        "--training-data-dir",
        type=str,
        default=None,
        help="training data directory",
    )
    parser.add_argument(
        "--pos-class-weight",
        type=float,
        default=1.0,
        help="positive class weight",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=Path(__file__).parent / "watermark_detector_mobilevit",
        help="save directory",
    )
    args = parser.parse_args()
    train(args)
    evaluate(args)
