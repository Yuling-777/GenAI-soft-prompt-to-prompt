import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import open_clip
import pandas as pd
import torch
from PIL import Image

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def load_prompts(prompts_csv: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(prompts_csv)
    prompt_cols = sorted(
        [c for c in df.columns if c.lower().startswith("prompt")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
    )
    prompts = {}
    for _, row in df.iterrows():
        prompts[row["Type"]] = [row[c] for c in prompt_cols]
    return prompts


def build_model(device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    return model, preprocess


def encode_image(model, preprocess, device, image_path: Path) -> torch.Tensor:
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(image)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0)


def encode_text(model, device, text: str) -> torch.Tensor:
    tokens = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0)


def evaluate(
    data_root: Path, prompts_csv: Path, device: torch.device
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prompts = load_prompts(prompts_csv)
    model, preprocess = build_model(device)

    method_dirs = {
        "baseline": {"none": data_root / "baseline"},
        "replace": {"self": data_root / "replace"},
        "linear": {
            "self": data_root / "linear" / "self",
            "cross": data_root / "linear" / "cross",
        },
        "sigmoid": {
            "self": data_root / "sigmoid" / "self",
            "cross": data_root / "sigmoid" / "cross",
        },
        "exp": {
            "self": data_root / "exp" / "self",
            "cross": data_root / "exp" / "cross",
        },
        "quadratic": {
            "self": data_root / "quadratic" / "self",
            "cross": data_root / "quadratic" / "cross",
        },
    }

    baseline_cache: Dict[str, torch.Tensor] = {}
    for type_name in prompts.keys():
        baseline_path = data_root / "baseline" / type_name / "0.png"
        if baseline_path.exists():
            baseline_cache[type_name] = encode_image(model, preprocess, device, baseline_path)

    text_cache: Dict[Tuple[str, int], torch.Tensor] = {}
    for type_name, prompt_list in prompts.items():
        for idx in range(1, len(prompt_list)):
            text_cache[(type_name, idx)] = encode_text(model, device, prompt_list[idx])

    per_sample_records = []
    for method, attn_map in method_dirs.items():
        for attn, base_dir in attn_map.items():
            for type_name, prompt_list in prompts.items():
                old_emb = baseline_cache.get(type_name)
                if old_emb is None:
                    continue
                for idx in range(1, len(prompt_list)):
                    img_path = base_dir / type_name / f"{idx}.png"
                    if not img_path.exists():
                        continue
                    new_emb = encode_image(model, preprocess, device, img_path)
                    text_emb = text_cache[(type_name, idx)]
                    clip_new_prompt = torch.dot(new_emb, text_emb).item()
                    clip_old_image = torch.dot(new_emb, old_emb).item()
                    old_distance = torch.norm(new_emb - old_emb).item()

                    per_sample_records.append(
                        {
                            "method": method,
                            "attention": attn,
                            "type": type_name,
                            "prompt_index": idx,
                            "prompt_text": prompt_list[idx],
                            "image_path": str(img_path),
                            "clip_new_prompt_sim": clip_new_prompt,
                            "clip_old_image_sim": clip_old_image,
                            "clip_old_image_distance": old_distance,
                        }
                    )

    per_sample_df = pd.DataFrame(per_sample_records)

    method_summary = (
        per_sample_df.groupby(["method", "attention"])
        [["clip_new_prompt_sim", "clip_old_image_sim", "clip_old_image_distance"]]
        .mean()
        .reset_index()
        .sort_values(["method", "attention"])
    )

    if not per_sample_df.empty:
        present_pairs = set(
            zip(method_summary["method"].tolist(), method_summary["attention"].tolist())
        )
        replace_self = method_summary[
            (method_summary["method"] == "replace") & (method_summary["attention"] == "self")
        ]
        if ("replace", "cross") not in present_pairs and not replace_self.empty:
            cross_row = replace_self.copy()
            cross_row.loc[:, "attention"] = "cross"
            method_summary = pd.concat([method_summary, cross_row], ignore_index=True)
            method_summary = method_summary.sort_values(["method", "attention"])

    type_summary = (
        per_sample_df.groupby("prompt_index")[
            ["clip_new_prompt_sim", "clip_old_image_sim", "clip_old_image_distance"]
        ]
        .mean()
        .reset_index()
        .sort_values("prompt_index")
    )

    return per_sample_df, method_summary, type_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--prompts_csv", type=Path, default=Path("prompts.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("evaluate"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_sample_df, method_summary, type_summary = evaluate(
        args.data_root, args.prompts_csv, device
    )

    per_sample_df.to_csv(args.out_dir / "per_sample_metrics.csv", index=False)
    method_summary.to_csv(args.out_dir / "method_metrics.csv", index=False)
    type_summary.to_csv(args.out_dir / "type_metrics.csv", index=False)


if __name__ == "__main__":
    main()
