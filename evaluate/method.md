# Metric Computation Methodology

- **Model/embeddings**: `open_clip` ViT-B/32 (`laion2b_s34b_b79k`). Images use the model-provided preprocessing; text is tokenized with `open_clip.tokenize`.
- **Data mapping**: For each `Type` in `prompts.csv`, `idx=0` is the original image (`data/baseline/<type>/0.png`). Edited images live in `data/{method}/{attention?}/{type}/{idx}.png` where `idx=1..10` correspond to `Prompt2…Prompt11` in the CSV. For `replace`, only `data/replace/<type>/` exists; its numbers are treated as `self` and duplicated for `cross` in aggregates.

## Per-sample metrics (`per_sample_metrics.csv`)
For every edited image:
- `clip_new_prompt_sim` = cosine similarity between the edited image embedding `E_edit` and the corresponding target prompt embedding `T_new` (matching its prompt index).\
  `cos(E_edit, T_new) = (E_edit · T_new) / (||E_edit|| * ||T_new||)`
- `clip_old_image_sim` = cosine similarity between the edited image embedding `E_edit` and the old image embedding `E_old` from `idx=0` of the same type.\
  `cos(E_edit, E_old)`
- `clip_old_image_distance` = Euclidean distance between the **L2-normalized** edited and old image embeddings.\
  `||Ē_edit – Ē_old||_2`

## Aggregations
- `method_metrics.csv`: mean of the three metrics grouped by `method` and `attention` (baseline uses `attention=none`; replace has `self` plus duplicated `cross` as noted).
- `type_metrics.csv`: mean of the three metrics grouped by `prompt_index` (1–10), averaging across all methods/attentions that provided that index.
