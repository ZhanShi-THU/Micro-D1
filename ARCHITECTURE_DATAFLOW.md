# Architecture Data Flow

This document describes the current modular VLM path implemented in the repository. It is intentionally limited to the active code path:

- `DINOv3 ViT-L/16 backbone`
- `dinotxt vision alignment head`
- `token-wise adapter`
- `Qwen3-VL text backbone`

The official Qwen3-VL vision tower is not used in this modular path.

## Input Records

The training and evaluation loaders consume unified JSONL manifest records. The fields used by the active multiple-choice path are:

- `image_path`
- `question`
- `choices`
- `correct_index`
- `target_text`
- `sample_id`
- `source_dataset`
- `split`

At runtime, the dataset loader converts each record into:

- a `PIL.Image`
- a multiple-choice prompt string
- a target string such as `The answer is (2)`

## Image Preprocessing

The current preprocessing path is:

```text
PIL.Image
-> Resize((448, 448))
-> ToTensor()
-> Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
-> pixel_values: [B, 3, 448, 448]
```

Why the fixed resize exists:

- batching requires a fixed image tensor shape
- the ViT patch grid must stay bounded so the visual prefix length is predictable
- the downstream Qwen decoder attention cost depends directly on the number of visual tokens

Tradeoff:

- forcing microscopy images to `448 x 448` can discard fine-grained details
- non-square images are geometrically stretched by the current implementation
- this is the main known visual-fidelity compromise in the current code path

## Vision Flow

### 1. DINOv3 backbone

The current backbone is `DINOv3 ViT-L/16`.

Relevant backbone properties:

- patch size: `16`
- hidden size: `1024`
- register/storage tokens: `4`

For the active input resolution `448 x 448`:

- `448 / 16 = 28`
- patch grid = `28 x 28`
- patch token count = `784`

The current implementation uses only patch tokens:

```text
pixel_values: [B, 3, 448, 448]
-> DINOv3 forward_features()
-> x_norm_patchtokens: [B, 784, 1024]
```

Notes:

- `x_storage_tokens` has shape `[B, 4, 1024]`
- those storage/register tokens are not used in the current modular VLM path
- if the backbone does not expose `x_norm_patchtokens`, the fallback path slices patch tokens out of `x_prenorm` while skipping the class token and any storage/register tokens

### 2. dinotxt alignment head

The dinotxt visual head keeps the token count unchanged and preserves the hidden width:

```text
[B, 784, 1024]
-> dinotxt head
-> [B, 784, 1024]
```

Semantically, this step pulls the DINO patch features toward a text-aligned semantic space before they are projected into Qwen's embedding space.

### 3. Adapter

The adapter is a token-wise two-layer MLP:

```text
LayerNorm(1024)
-> Linear(1024 -> 2048)
-> GELU
-> Linear(2048 -> 4096)
-> Dropout
```

So the visual output becomes:

```text
aligned visual tokens: [B, 784, 1024]
-> adapter
-> visual_embeds: [B, 784, 4096]
```

## Text Flow

The multiple-choice prompt and target answer are tokenized by the Qwen tokenizer.

Current text settings:

- `max_text_length = 512`
- pad token defaults to EOS if the tokenizer does not define one

For a batch:

```text
input_ids: [B, T]
attention_mask: [B, T]
labels: [B, T]
```

where `T <= 512`.

The text embeddings come from the Qwen3-VL text embedding layer:

```text
input_ids: [B, T]
-> embed_tokens
-> text_embeds: [B, T, 4096]
```

## Multimodal Fusion

The current fusion strategy is prefix concatenation only.

```text
visual_embeds: [B, 784, 4096]
text_embeds: [B, T, 4096]
-> concat on sequence axis
-> inputs_embeds: [B, 784 + T, 4096]
```

The attention mask is expanded accordingly:

```text
visual_attention_mask: [B, 784] filled with 1
text attention_mask: [B, T]
-> merged_attention_mask: [B, 784 + T]
```

The labels are also expanded so the visual prefix does not contribute to the autoregressive loss:

```text
visual_ignore: [B, 784] filled with -100
text labels: [B, T]
-> merged labels: [B, 784 + T]
```

This means the visual tokens participate in attention, but not in direct token-level supervision.

## DeepStack Injection

In addition to prefix concatenation, the current implementation also supports a shared DeepStack injection path into the Qwen3-VL text backbone.

The design is intentionally minimal:

- the project reuses the same adapter output that is already used for the visual prefix
- no per-layer projection is added
- no gate is added
- the same visual embedding tensor is reused for the first several decoder layers

With the current default config:

- `use_deepstack_injection = true`
- `deepstack_num_layers = 4`

The extra tensors are:

```text
visual_pos_masks: [B, visual_len + T]
deepstack_visual_embeds: list of length 4
```

The semantics are:

- `visual_pos_masks[:, :visual_len] = True`
- `visual_pos_masks[:, visual_len:] = False`
- each list element in `deepstack_visual_embeds` is the same shared adapter output flattened over all visual positions

For example, with `224 x 224` inputs during smoke testing:

- visual prefix length = `14 x 14 = 196`
- `visual_pos_masks` has shape `[B, 196 + T]`
- each `deepstack_visual_embeds[i]` has shape `[B * 196, 4096]`

For the full `448 x 448` training path:

- visual prefix length = `28 x 28 = 784`
- `visual_pos_masks` has shape `[B, 784 + T]`
- each `deepstack_visual_embeds[i]` has shape `[B * 784, 4096]`

Conceptually, the full visual path is now:

```text
DINOv3
-> dinotxt head
-> adapter
-> visual prefix embeddings
-> prefix concat into inputs_embeds
```

and in parallel:

```text
adapter output
-> shared DeepStack list
-> injected into the first 4 Qwen decoder layers
```

So the current model uses two complementary injection routes:

- prefix injection makes visual tokens part of the unified input sequence
- DeepStack injection re-injects the same visual semantics into the early hidden states as residual additions

This is not a reproduction of the official Qwen3-VL visual tower. The project does not extract multi-layer visual features from the official Qwen vision encoder. Instead, it reuses the project’s own adapter output as a shared DeepStack signal.

## Qwen3-VL Text Backbone

The active language model path uses the text backbone of `Qwen3-VL-8B-Instruct`.

Relevant text-side properties:

- text hidden size: `4096`
- decoder layers: `36`

The current forward path is:

```text
inputs_embeds: [B, 784 + T, 4096]
attention_mask: [B, 784 + T]
-> Qwen3-VL text backbone
-> hidden_states: [B, 784 + T, 4096]
```

Then the language head projects to vocabulary logits:

```text
hidden_states: [B, 784 + T, 4096]
-> lm_head
-> logits: [B, 784 + T, vocab_size]
```

For the current local Qwen3-VL-8B checkpoint:

- vocab size = `151936`

So the logits shape is:

```text
[B, 784 + T, 151936]
```

## Loss

The training loss is standard autoregressive next-token cross-entropy over the merged sequence:

- logits are shifted left by one position
- labels are shifted right by one position
- positions with label `-100` are ignored

Because the visual prefix labels are all `-100`, the model is not trained to predict visual tokens. Instead, it is trained so that the visual prefix helps the Qwen text backbone predict the answer tokens.

## Interpretation

The active training objective is not trying to teach the model visual semantics from scratch.

The division of labor is:

- `DINOv3`: dense fine-grained visual representation
- `dinotxt head`: pull visual tokens toward a text-compatible semantic space
- `adapter`: map those aligned visual tokens into Qwen's text embedding space
- `Qwen3-VL text backbone`: consume the visual prefix and autoregressively produce the answer text

In short:

- the vision model is responsible for seeing
- the alignment head is responsible for semantic bridging
- the adapter is responsible for space matching
- the language model is responsible for reasoning and decoding
