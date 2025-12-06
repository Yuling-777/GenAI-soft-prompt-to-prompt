import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
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
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device)
    return model, preprocess


def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # remove redundant dim
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor:
    BLOCK_KEY = "block"
    ATTN_KEY = "attn"
    PATCH_IMD_KEY = "patch_imd"
    QKV_KEY = "qkv"
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load("facebookresearch/dino:main", model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = list(range(12))
        self.layers_dict[VitExtractor.ATTN_KEY] = list(range(12))
        self.layers_dict[VitExtractor.QKV_KEY] = list(range(12))
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = list(range(12))
        for key in VitExtractor.KEY_LIST:
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def _run_and_collect(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = 8 if "8" in self.model_name else 16
        return 1 + (h // patch_size * w // patch_size)

    def get_head_num(self):
        return 6 if "s" in self.model_name else 12

    def get_embedding_dim(self):
        return 384 if "s" in self.model_name else 768

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map


class DinoStructureLoss:
    def __init__(self, device: torch.device):
        self.extractor = VitExtractor(model_name="dino_vitb8", device=device)
        self.preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.device = device

    def get_self_sim(self, tensor_img: torch.Tensor):
        return self.extractor.get_keys_self_sim_from_input(tensor_img, layer_num=11)

    def struct_distance(self, base_self_sim: torch.Tensor, edit_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            keys_ssim = self.get_self_sim(edit_tensor)
            return F.mse_loss(keys_ssim, base_self_sim).item()


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
    dino_loss = DinoStructureLoss(device)

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
    baseline_dino_cache: Dict[str, torch.Tensor] = {}
    for type_name in prompts.keys():
        baseline_path = data_root / "baseline" / type_name / "0.png"
        if baseline_path.exists():
            baseline_cache[type_name] = encode_image(model, preprocess, device, baseline_path)
            pil_base = Image.open(baseline_path).convert("RGB")
            base_tensor = dino_loss.preprocess(pil_base).unsqueeze(0).to(device)
            baseline_dino_cache[type_name] = dino_loss.get_self_sim(base_tensor)

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

                    pil_edit = Image.open(img_path).convert("RGB")
                    edit_tensor = dino_loss.preprocess(pil_edit).unsqueeze(0).to(device)
                    dino_distance = dino_loss.struct_distance(
                        baseline_dino_cache[type_name], edit_tensor
                    )

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
                            "dino_struct_dist": dino_distance,
                        }
                    )

    per_sample_df = pd.DataFrame(per_sample_records)

    method_summary = (
        per_sample_df.groupby(["method", "attention"])
        [["clip_new_prompt_sim", "clip_old_image_sim", "dino_struct_dist"]]
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
            ["clip_new_prompt_sim", "clip_old_image_sim", "dino_struct_dist"]
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
