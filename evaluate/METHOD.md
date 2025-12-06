# Metric Computation Methodology

- **Model/embeddings**: `open_clip` ViT-B/32 (`openai` weights). Image/text similarities use raw cosine (embeddings are L2-normalized). Images are preprocessed with the model defaults; text is tokenized with `open_clip.tokenize`.
- **Data mapping**: For each `Type` in `prompts.csv`, `idx=0` is the original image (`data/baseline/<type>/0.png`). Edited images live in `data/{method}/{attention?}/{type}/{idx}.png` where `idx=1..10` correspond to `Prompt2…Prompt11` in the CSV. For `replace`, only `data/replace/<type>/` exists; its numbers are treated as `self` and duplicated for `cross` in aggregates.
- **DINO structure model**: ViT-B/8 from `facebookresearch/dino` (via `torch.hub`). Structure distance compares self-similarity maps of attention keys at layer 11 after resizing images to 224 and applying ImageNet normalization.

## Per-sample metrics (`per_sample_metrics.csv`)
For every edited image:
- `clip_new_prompt_sim` = cosine similarity between the edited image embedding `E_edit` and the target prompt embedding `T_new`:\
  `cos(E_edit, T_new)`
- `clip_old_image_sim` = cosine similarity between the edited image embedding `E_edit` and the old image embedding `E_old` (`idx=0` of the same type):\
  `cos(E_edit, E_old)`
- `dino_struct_dist` = MSE between the DINO key self-similarity map of the edited image and that of the baseline image for the same type:\
  `MSE(SSIM_edit, SSIM_base)` where `SSIM` is computed from attention-key cosine similarities at layer 11 (per `dino_struct_dist.py` logic).

## Aggregations
- `method_metrics.csv`: mean of the three metrics grouped by `method` and `attention` (baseline uses `attention=none`; replace has `self` plus duplicated `cross` as noted).
- `type_metrics.csv`: mean of the three metrics grouped by `prompt_index` (1–10), averaging across all methods/attentions that provided that index.
