"""Recreate BeeBeeOmni SGLang image preprocessing for a saved request."""

import argparse
import base64
import io
import json
from collections import defaultdict

import torch
from PIL import Image

from sglang.srt.multimodal.image_processor_opt import Qwen25VLImageProcessorOptimized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request_json")
    parser.add_argument("model_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    request = json.load(open(args.request_json, encoding="utf-8"))
    images = [
        Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        for data in request["image_data"]
    ]
    pairs = [images[i : i + 2] for i in range(0, len(images), 2)]
    buckets = defaultdict(list)
    for pair_idx, pair in enumerate(pairs):
        buckets[pair[0].size].append((pair_idx, pair))

    processor = Qwen25VLImageProcessorOptimized.from_pretrained(args.model_path)
    ordered_pixels = [None] * len(pairs)
    ordered_grids = [None] * len(pairs)
    for bucket in buckets.values():
        batch_images = [image for _, pair in bucket for image in pair]
        outputs = processor.preprocess(
            batch_images, return_tensors="pt", patch_reshape_method="torch"
        )
        features = outputs["pixel_values"]
        grid_h, grid_w = map(int, outputs["grid_thw"][0, 1:].tolist())
        stride = grid_h * grid_w
        for offset, (pair_idx, _) in enumerate(bucket):
            ordered_pixels[pair_idx] = features[offset * stride : (offset + 1) * stride]
            ordered_grids[pair_idx] = torch.tensor([[1, grid_h, grid_w]])

    result = {
        "pixel_values": torch.cat(ordered_pixels),
        "grid_thw": torch.cat(ordered_grids),
    }
    torch.save(result, args.output_path)
    print(
        f"saved pixel_values={tuple(result['pixel_values'].shape)} "
        f"grid_thw={tuple(result['grid_thw'].shape)}"
    )


if __name__ == "__main__":
    main()
