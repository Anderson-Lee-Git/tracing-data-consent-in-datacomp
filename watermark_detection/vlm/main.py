# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
import argparse
import json
import textwrap
import os

load_dotenv()


def load_image(path: Path):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    # increase resolution while retaining aspect ratio
    width, height = image.size
    max_dim = 640
    if width > height:
        image = image.resize(
            (max_dim, int(max_dim * height / width)), Image.Resampling.LANCZOS
        )
    else:
        image = image.resize(
            (int(max_dim * width / height), max_dim), Image.Resampling.LANCZOS
        )
    return image


def create_batch_messages(image_list):
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img},
                    {
                        "type": "text",
                        "text": textwrap.dedent(
                            """
                            A watermark on an image is a deliberately embedded visual marker—often semi-transparent text, logos,or patterns—designed to assert ownership, deter unauthorized use, or signal authenticity. It can also be a form of a link, brand name, or author name at the top/bottom corner of the image. Does this image contain any watermark? If so, return the text of the watermark. Otherwise, return no in lowercase.
                            """
                        ),
                    },
                ],
            }
        ]
        for img in image_list
    ]


def process_batch(image_paths, processor, model, batch_size=8):
    results = {"path": [], "text": []}

    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]

        # Load batch of images
        batch_images = [load_image(path) for path in batch_paths]

        # Create batch messages
        messages = create_batch_messages(batch_images)
        templated_messages = []
        # Process batch
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
        ).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        batch_str_paths = [str(path) for path in batch_paths]
        results["path"].extend(batch_str_paths)
        results["text"].extend(output_text)

        # Clean up batch
        for img in batch_images:
            img.close()
        # Clean up tensors
        del inputs, generated_ids, generated_ids_trimmed
    return results


def main(args):
    cache_dir = os.getenv("TRANSFORMERS_CACHE")
    watermark_image_paths = (
        list(Path(args.watermark_image_dir).glob("*.jpg"))
        + list(Path(args.watermark_image_dir).glob("*.jpeg"))
        + list(Path(args.watermark_image_dir).glob("*.png"))
    )
    clean_image_paths = (
        list(Path(args.clean_image_dir).glob("*.jpg"))
        + list(Path(args.clean_image_dir).glob("*.jpeg"))
        + list(Path(args.clean_image_dir).glob("*.png"))
    )
    image_paths = watermark_image_paths + clean_image_paths
    print(f"Length of watermark image paths: {len(watermark_image_paths)}")
    print(f"Length of clean image paths: {len(clean_image_paths)}")
    print(f"Length of image paths: {len(image_paths)}")
    # Load model
    if args.model_type == "rolm-ocr":
        ckpt = "reducto/RolmOCR"
    elif args.model_type == "gemma":
        ckpt = "google/gemma-3-12b-it"
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    processor = AutoProcessor.from_pretrained(ckpt, cache_dir=cache_dir)
    model = (
        AutoModelForImageTextToText.from_pretrained(ckpt, cache_dir=cache_dir)
        .eval()
        .to("cuda")
    )
    results = process_batch(
        image_paths, model=model, processor=processor, batch_size=args.batch_size
    )
    with open(f"{args.model_type}-results.json", "w") as f:
        json.dump(results, f)
    with open(f"{args.model_type}-results.json", "r") as f:
        results = json.load(f)
    labels = [1] * len(watermark_image_paths) + [0] * len(clean_image_paths)
    preds = []
    print(f"Length of result texts: {len(results['text'])}")
    # Print results
    for path, text in zip(results["path"], results["text"]):
        if text.lower().startswith("no"):
            preds.append(0)
        else:
            preds.append(1)
    labels = np.array(labels)
    preds = np.array(preds)
    print(f"Accuracy: {np.sum(preds == labels) / len(preds)}")
    print(f"Precision: {np.sum((preds == 1) & (labels == 1)) / np.sum(preds == 1)}")
    print(f"Recall: {np.sum((preds == 1) & (labels == 1)) / np.sum(labels == 1)}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--watermark-image-dir", type=str, default=None)
    argparser.add_argument("--clean-image-dir", type=str, default=None)
    argparser.add_argument("--batch-size", type=int, default=8)
    argparser.add_argument("--model-type", type=str, choices=["rolm-ocr", "gemma"])
    args = argparser.parse_args()
    main(args)
