from io import BytesIO
import os
import argparse
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import numpy as np
from torch.utils.data import Dataset
from webdataset import WebDataset
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Add this near the top of the file, after imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Segmentation class
class YOLOSEG:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        height, width, _ = img.shape
        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        if len(result) > 0:
            for seg in result.masks.xy:
                segment = np.array(seg, dtype=np.float32)
                segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contours_idx, scores


class ImagePathDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        # increase resolution while maintaining aspect ratio
        width, height = pil_image.size
        if width > height:
            pil_image = pil_image.resize(
                (1024, int(1024 * height / width)), Image.Resampling.LANCZOS
            )
        else:
            pil_image = pil_image.resize(
                (int(1024 * width / height), 1024), Image.Resampling.LANCZOS
            )
        cv_image = cv2.imread(str(image_path))
        # increase resolution while maintaining aspect ratio
        width, height = cv_image.shape[:2]
        if width > height:
            cv_image = cv2.resize(
                cv_image,
                (1024, int(1024 * height / width)),
                interpolation=cv2.INTER_LANCZOS4,
            )
        else:
            cv_image = cv2.resize(
                cv_image,
                (int(1024 * width / height), 1024),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return {
            "pil_image": pil_image,
            "cv_image": cv_image,
            "image_path": image_path,
        }


# Function to estimate text size
def estimate_text_size(label, font_size):
    approx_char_width = font_size * 0.6
    text_width = len(label) * approx_char_width
    text_height = font_size
    return text_width, text_height


def write_detections_to_file(
    image_path, detections, detection_dir, detection_prefix, detection_suffix
):
    if len(detections) == 0:
        return
    # Create a text file named after the image
    text_file_path = (
        detection_dir / f"{detection_prefix}{image_path.stem}{detection_suffix}.txt"
    )
    with open(text_file_path, "w") as file:
        for detection in detections:
            file.write(f"{detection}\n")


def load_pil_image(jpg):
    im = Image.open(BytesIO(jpg))
    if im.mode != "RGB":
        im = im.convert("RGB")
    # increase resolution
    im = im.resize((1024, 1024), Image.Resampling.LANCZOS)
    return im


def load_cv_image(jpg):
    # Convert bytes to numpy array
    nparr = np.frombuffer(jpg, np.uint8)
    # Decode the numpy array as an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # increase resolution
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
    return img


def load_shard_id(sample):
    shard_id = sample["__local_path__"].split("/")[-1].split(".")[0]
    return shard_id


def load_data_scale(sample):
    data_scale = sample["__local_path__"].split("/")[-3]
    return data_scale


def load_image_id(sample):
    shard_id = load_shard_id(sample)
    image_id = str(sample["__key__"])
    image_id = image_id[len(shard_id) :]
    return image_id


def load_sample(sample):
    json_data = json.loads(sample["json"])
    return {
        "pil_image": load_pil_image(sample["jpg"]),
        "cv_image": load_cv_image(sample["jpg"]),
        "url": json_data["url"],
        "exif": json_data["exif"],
        "uid": json_data["uid"],
        "caption": json_data["caption"],
        "shard_id": load_shard_id(sample),
        "data_scale": load_data_scale(sample),
        "image_id": load_image_id(sample),
    }


def filter_success_sample(sample):
    json_data = json.loads(sample["json"])
    return json_data["status"] == "success"


def get_dataset(args):
    if args.data_format == "directory":
        # Store input images in a variable
        input_dir = Path(args.input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        for extension in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(input_dir.rglob(extension))
        return ImagePathDataset(image_paths)
    elif args.data_format == "webdataset":
        input_webdataset_file = Path(args.input_webdataset_file)
        assert (
            input_webdataset_file.exists()
        ), f"Input directory {input_webdataset_file} does not exist"
        assert args.input_webdataset_file.endswith(
            ".tar"
        ), f"Input directory {args.input_webdataset_file} must end with .tar"
        dataset = WebDataset(
            args.input_webdataset_file,
            cache_dir="/gscratch/scrubbed/lee0618/cache",
            shardshuffle=False,
        )
        dataset = dataset.select(filter_success_sample)
        dataset = dataset.map(load_sample)
        return dataset
    else:
        raise ValueError(f"Invalid data format: {args.data_format}")


def generate_detection(
    dataset,
    model,
    ys,
    mode,
    selected_classes,
    class_overrides,
    confidence_threshold,
    label_boxes,
    font_size,
    class_colors,
    text_colors,
    detect_all_classes,
    overlay_dir,
    detection_dir,
    mask_dir,
    overlay_prefix,
    overlay_suffix,
    mask_prefix,
    mask_suffix,
    detection_prefix,
    detection_suffix,
    font,
    args,
):
    logger.info(f"Generating outputs in {mode} mode.")
    counter = 0
    for sample in tqdm(dataset, desc="Processing Images"):
        counter += 1
        img_pil = sample["pil_image"]
        img_cv = sample["cv_image"]
        image_path = None
        if args.data_format == "webdataset":
            output_dir = Path(args.output_dir)
            image_dir = output_dir / "images"
            shard_dir = image_dir / sample["data_scale"] / sample["shard_id"]
            image_path = shard_dir / (sample["image_id"] + ".jpg")
        else:
            image_path = sample["image_path"]
        assert image_path is not None, "Image path is not found"
        # Detection Mode
        if mode == "detection":
            mask_img = np.zeros(
                img_cv.shape[:2], dtype=np.uint8
            )  # Initialize a blank mask for all detections
            results = model.predict(img_pil)
            draw = ImageDraw.Draw(img_pil)
            detections = []

            if len(results) > 0 and results[0].boxes.xyxy is not None:
                for idx, box in enumerate(results[0].boxes.xyxy):
                    x1, y1, x2, y2 = box[:4].tolist()
                    cls_id = int(results[0].boxes.cls[idx].item())
                    conf = results[0].boxes.conf[idx].item()
                    cls_name = (
                        results[0].names[cls_id]
                        if 0 <= cls_id < len(results[0].names)
                        else "Unknown"
                    )
                    cls_name = class_overrides.get(cls_name, cls_name)

                    if (
                        cls_name in selected_classes or detect_all_classes
                    ) and conf >= confidence_threshold:
                        box_color = class_colors.get(cls_name, (255, 0, 0))
                        text_color = text_colors.get(cls_name, "black")
                        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=7)

                        # Fill mask image for this detection
                        cv2.rectangle(
                            mask_img,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            255,
                            thickness=-1,
                        )  # -1 thickness fills the rectangle

                        if label_boxes:
                            label = f"{cls_name}: {conf:.2f}"
                            text_size = estimate_text_size(label, font_size)
                            draw.rectangle(
                                [x1, y1 - text_size[1] - 5, x1 + text_size[0], y1],
                                fill=box_color,
                            )
                            draw.text(
                                (x1, y1 - text_size[1] - 5),
                                label,
                                fill=text_color,
                                font=font,
                            )

                        # Add detection data to the list
                        detections.append(f"{cls_name} {conf:.2f} {x1} {y1} {x2} {y2}")

            # Save overlay images
            if len(detections) > 0 and args.output_overlay_image:
                if args.data_format == "directory":
                    img_pil.save(
                        overlay_dir
                        / ("clean" if "clean" in str(image_path) else "watermark")
                        / f"{overlay_prefix}{image_path.stem}{overlay_suffix}{image_path.suffix}"
                    )
                else:
                    img_pil.save(
                        overlay_dir
                        / f"{overlay_prefix}{image_path.stem}{overlay_suffix}{image_path.suffix}"
                    )

            # Write detections to a text file
            if len(detections) > 0 and args.output_text:
                if args.data_format == "directory":
                    write_detections_to_file(
                        image_path,
                        detections,
                        detection_dir
                        / ("clean" if "clean" in str(image_path) else "watermark"),
                        detection_prefix,
                        detection_suffix,
                    )
                else:
                    write_detections_to_file(
                        image_path,
                        detections,
                        detection_dir,
                        detection_prefix,
                        detection_suffix,
                    )
            # Save the combined mask image
            mask_output_path = (
                mask_dir / f"{mask_prefix}{image_path.stem}{mask_suffix}.png"
            )
            if len(detections) > 0 and args.output_mask_image:
                cv2.imwrite(str(mask_output_path), mask_img)
        elif mode == "segmentation":
            height, width, _ = img_cv.shape

            # Perform inference using YOLOSEG for segmentation masks
            bboxes, classes, segmentations, scores = ys.detect(img_cv)

            # Initialize a blank mask for all segmentations
            mask_img = np.zeros(img_cv.shape[:2], dtype=np.uint8)

            # Perform inference using the original YOLO model for initial annotation
            results = model.predict(img_pil)
            if hasattr(results[0], "render"):
                annotated_img = results[0].render()[0]  # Use 'render' if available
            else:
                annotated_img = results[0].plot()  # Use 'plot' as a fallback
            annotated_img = np.array(
                annotated_img
            )  # Convert PIL image to NumPy array for CV2 processing

            # Text file for saving segmentation data
            txt_output_path = (
                detection_dir
                / f"{detection_prefix}{image_path.stem}{detection_suffix}.txt"
            )
            if args.output_text:
                with open(txt_output_path, "w") as f:
                    for bbox, class_id, seg in zip(bboxes, classes, segmentations):
                        # Normalize the segmentation data
                        seg_normalized = seg / [width, height]
                        # Write normalized data to text file
                        seg_data = " ".join(
                            [f"{x:.6f},{y:.6f}" for x, y in seg_normalized]
                        )
                        f.write(f"{class_id} {seg_data}\n")

                        # Draw segmentation mask on the combined mask image
                        cv2.fillPoly(mask_img, [np.array(seg, dtype=np.int32)], 255)

                        # Draw bounding box and segmentation mask on the annotated image
                        x, y, x2, y2 = bbox
                        cv2.rectangle(annotated_img, (x, y), (x2, y2), (0, 0, 255), 2)
                        cv2.polylines(
                            annotated_img,
                            [np.array(seg, dtype=np.int32)],
                            isClosed=True,
                            color=(0, 0, 255),
                            thickness=2,
                        )

            # Save the final annotated image with bounding boxes and segmentation masks
            overlay_output_path = (
                overlay_dir
                / f"{overlay_prefix}{image_path.stem}{overlay_suffix}{image_path.suffix}"
            )
            if args.output_overlay_image:
                cv2.imwrite(str(overlay_output_path), annotated_img)

            # Save the combined mask image
            mask_output_path = (
                mask_dir / f"{mask_prefix}{image_path.stem}{mask_suffix}.png"
            )
            if args.output_mask_image:
                cv2.imwrite(str(mask_output_path), mask_img)
    logger.info(
        f"Processed {counter} images. Overlays saved to '{overlay_dir}', Detections saved to '{detection_dir}', and Masks saved to '{mask_dir}'."
    )


def eval_detection_results(dataset, detection_dir, detection_prefix, detection_suffix):
    labels = []
    preds = []
    results = {"path": [], "text": []}
    for sample in tqdm(dataset, desc="Summarizing Detection Results"):
        image_path = sample["image_path"]
        label = 1 if image_path.parent.name == "watermark" else 0
        text_file_path = (
            detection_dir
            / ("clean" if "clean" in str(image_path) else "watermark")
            / f"{detection_prefix}{image_path.stem}{detection_suffix}.txt"
        )
        if not text_file_path.exists():
            pred = 0
        else:
            with open(text_file_path, "r") as file:
                detections = file.readlines()
                pred = 1 if len(detections) > 0 else 0
        labels.append(label)
        preds.append(pred)
        if pred == 1:
            results["text"].append("yes")
        else:
            results["text"].append("no")
        results["path"].append(str(image_path))
    with open(os.getenv("WATERMARK_DETECTION_YOLO_RESULTS_PATH"), "w") as f:
        json.dump(results, f)
    labels = np.array(labels)
    preds = np.array(preds)
    logger.info(f"Evaluation Accuracy: {np.sum(labels == preds) / len(labels)}")
    if np.sum(preds == 1) == 0:
        logger.info("No positive predictions")
    else:
        logger.info(
            f"Evaluation Precision: {np.sum((preds == 1) & (labels == 1)) / np.sum(preds == 1)}"
        )
    if np.sum(labels == 1) == 0:
        logger.info("No positive labels")
    else:
        logger.info(
            f"Evaluation Recall: {np.sum((preds == 1) & (labels == 1)) / np.sum(labels == 1)}"
        )


def main(args):
    if args.data_format == "webdataset":
        if args.output_dir is None:
            output_dir = Path(os.getenv("WATERMARK_DETECTION_OUTPUT_DIR"))
        else:
            output_dir = Path(args.output_dir)
        data_scale = args.input_webdataset_file.split("/")[-3]
        shard_id = args.input_webdataset_file.split("/")[-1].split(".")[0]
        output_dir = output_dir / data_scale / shard_id
    else:
        if args.output_dir is None:
            args.output_dir = "./generate_output"
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.data_format == "directory":
        (overlay_dir / "watermark").mkdir(parents=True, exist_ok=True)
        (overlay_dir / "clean").mkdir(parents=True, exist_ok=True)
    overlay_prefix = ""  # Image prefix, can be empty
    overlay_suffix = ""  # Image suffix, can be empty

    detection_dir = output_dir / "detections"
    detection_dir.mkdir(parents=True, exist_ok=True)
    if args.data_format == "directory":
        (detection_dir / "watermark").mkdir(parents=True, exist_ok=True)
        (detection_dir / "clean").mkdir(parents=True, exist_ok=True)
    detection_prefix = ""  # Text prefix, can be empty
    detection_suffix = ""  # Text suffix, can be empty

    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_prefix = ""  # Text prefix, can be empty
    mask_suffix = ""  # Text suffix, can be empty

    # Load your trained model
    model_path = "./models/best.pt"
    model = YOLO(model_path)

    # Mode selection: detection or segmentation
    mode = "detection"

    # Detect all classes or selected classes only
    detect_all_classes = True  # Set to True to detect all classes, False to detect only specific classes below

    # Classes to detect
    # Example: ['SpeechBalloons', 'General_speech', 'hit_sound', 'blast_sound', 'narration speech', 'thought_speech', 'roar']
    selected_classes = ["socks"]

    # Class override mapping, treats the left side of the mapping as if it was the class of the right side
    # Example: thought_speech annotations will be treated as SpeechBalloons annotations.
    class_overrides = {
        "thought_speech": "SpeechBalloons",
    }

    # Confidence threshold
    confidence_threshold = args.confidence

    # Label settings
    label_boxes = True  # Draw class names or just boxes
    font_size = 30  # Font size for the class labels

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
        logger.info("Default font will be used, as custom font not found.")

    # Label colors by index
    predefined_colors_with_text = [
        ((204, 0, 0), "white"),  # Darker red, white text
        ((0, 204, 0), "black"),  # Darker green, black text
        ((0, 0, 204), "white"),  # Darker blue, white text
        ((204, 204, 0), "black"),  # Darker yellow, black text
        ((204, 0, 204), "white"),  # Darker magenta, white text
        ((0, 204, 204), "black"),  # Darker cyan, black text
        ((153, 0, 0), "white"),  # Darker maroon, white text
        ((0, 153, 0), "white"),  # Darker green, white text
        ((0, 0, 153), "white"),  # Darker navy, white text
        ((153, 153, 0), "black"),  # Darker olive, black text
        # Add more color pairs if needed
    ]

    # Assign colors to each class, wrapping around if there are more classes than colors
    class_colors = {
        class_name: predefined_colors_with_text[i % len(predefined_colors_with_text)][0]
        for i, class_name in enumerate(selected_classes)
    }
    text_colors = {
        class_name: predefined_colors_with_text[i % len(predefined_colors_with_text)][1]
        for i, class_name in enumerate(selected_classes)
    }

    dataset = get_dataset(args)

    ys = YOLOSEG(model_path)
    if args.eval_detection_results:
        # clear detection directory
        for file in detection_dir.glob("*.txt"):
            file.unlink()
        # clear mask directory
        for file in mask_dir.glob("*.png"):
            file.unlink()
        # clear overlay directory
        for file in overlay_dir.glob("*.jpg"):
            file.unlink()

    # Process images with progress bar
    if args.generate:
        generate_detection(
            dataset,
            model,
            ys,
            mode,
            selected_classes,
            class_overrides,
            confidence_threshold,
            label_boxes,
            font_size,
            class_colors,
            text_colors,
            detect_all_classes,
            overlay_dir,
            detection_dir,
            mask_dir,
            overlay_prefix,
            overlay_suffix,
            mask_prefix,
            mask_suffix,
            detection_prefix,
            detection_suffix,
            font,
            args,
        )

    if args.eval_detection_results:
        eval_detection_results(
            dataset, detection_dir, detection_prefix, detection_suffix
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data-format",
        type=str,
        choices=["directory", "webdataset"],
        default="directory",
    )
    argparser.add_argument("--confidence", type=float, default=0.15)
    argparser.add_argument("--input-dir", type=str, default="./generate_input")
    argparser.add_argument("--input-webdataset-file", type=str, default=None)
    argparser.add_argument("--output-dir", type=str, default=None)
    argparser.add_argument("--output-text", action="store_true")
    argparser.add_argument("--output-overlay-image", action="store_true")
    argparser.add_argument("--output-mask-image", action="store_true")
    argparser.add_argument("--generate", action="store_true")
    argparser.add_argument("--eval-detection-results", action="store_true")
    args = argparser.parse_args()
    main(args)
