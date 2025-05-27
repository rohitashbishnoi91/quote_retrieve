---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: quote
  sentences:
  - “i've been making a list of the things they don't teach you at school. they don't
    teach you how to love somebody. they don't teach you how to be famous. they don't
    teach you how to be rich or how to be poor. they don't teach you how to walk away
    from someone you don't love any longer. they don't teach you how to know what's
    going on in someone else's mind. they don't teach you what to say to someone who's
    dying. they don't teach you anything worth knowing.” - neil gaiman, (dying, facts,
    fame, knowing, love, poverty, reality, school, teach, wealth)
  - '“they say a person needs just three things to be truly happy in this world: someone
    to love, something to do, and something to hope for.” - tom bodett (happiness,
    hope, joy, love, wishing)'
  - “the scariest moment is always just before you start.” - stephen king, (writing)
- source_sentence: quote
  sentences:
  - “even if you cannot change all the people around you, you can change the people
    you choose to be around. life is too short to waste your time on people who donâ€™t
    respect, appreciate, and value you. spend your life with people who make you smile,
    laugh, and feel loved.” - roy t. bennett, (appreciate, change, happiness, inspiration,
    inspirational, inspirational-attitude, inspirational-life, inspirational-quotes,
    inspire, inspiring, laugh, life, life-and-living, life-lessons, life-quotes, living,
    love, optimism, optimistic, positive, positive-affirmation, positive-life, positive-thinking,
    relationship, respect, smile, value)
  - “all that is gold does not glitter,not all those who wander are lost;the old that
    is strong does not wither,deep roots are not reached by the frost.from the ashes
    a fire shall be woken,a light from the shadows shall spring;renewed shall be blade
    that was broken,the crownless again shall be king.” - j.r.r. tolkien, (frost,
    glitter, gold, lost, poetry, roots, strength, strong, wander, wither)
  - “never put off till tomorrow what may be done day after tomorrow just as well.”
    - mark twain (humor, procrastination)
- source_sentence: quote
  sentences:
  - “the story so far:in the beginning the universe was created.this has made a lot
    of people very angry and been widely regarded as a bad move.” - douglas adams,
    (humor, scifi)
  - “what you must understand about me is that iâ€™m a deeply unhappy person.” - john
    green, (sadness, unhappiness)
  - “where is human nature so weak as in the bookstore?” - henry ward beecherr (books,
    humor)
- source_sentence: quote
  sentences:
  - “let us read, and let us dance; these two amusements will never do any harm to
    the world.” - voltaire (dance, dancing, reading)
  - “the truth is, unless you let go, unless you forgive yourself, unless you forgive
    the situation, unless you realize that the situation is over, you cannot move
    forward.” - steve maraboli, (forgiveness, inspirational, letting-go, life, motivational,
    moving-forward, truth)
  - “knowing yourself is the beginning of all wisdom.” - aristotle (introspection,
    self-discovery, wisdom)
- source_sentence: quote
  sentences:
  - “what is a friend? a single soul dwelling in two bodies.” - aristotle (friendship,
    soul)
  - “don't justdon't just learn, experience.don't just read, absorb.don't just change,
    transform.don't just relate, advocate.don't just promise, prove.don't just criticize,
    encourage.don't just think, ponder.don't just take, give.don't just see, feel.donâ€™t
    just dream, do. don't just hear, listen.don't just talk, act.don't just tell,
    show.don't just exist, live.” - roy t. bennett, (act, action, change, criticize,
    doing, dream, dreams, encouragement, experience, feeling, giving, inspiration,
    inspirational, inspirational-attitude, inspirational-life, inspirational-quote,
    inspirational-quotes, inspire, inspiring, learning, life, life-and-living, life-lessons,
    life-philosophy, life-quotes, listening, living, optimism, optimistic, ponder,
    positive, positive-affirmation, positive-life, positive-thinking, thinking, transform)
  - “hell is empty and all the devils are here.” - william shakespeare, (inspirational)
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'quote',
    '“hell is empty and all the devils are here.” - william shakespeare, (inspirational)',
    "“don't justdon't just learn, experience.don't just read, absorb.don't just change, transform.don't just relate, advocate.don't just promise, prove.don't just criticize, encourage.don't just think, ponder.don't just take, give.don't just see, feel.donâ€™t just dream, do. don't just hear, listen.don't just talk, act.don't just tell, show.don't just exist, live.” - roy t. bennett, (act, action, change, criticize, doing, dream, dreams, encouragement, experience, feeling, giving, inspiration, inspirational, inspirational-attitude, inspirational-life, inspirational-quote, inspirational-quotes, inspire, inspiring, learning, life, life-and-living, life-lessons, life-philosophy, life-quotes, listening, living, optimism, optimistic, ponder, positive, positive-affirmation, positive-life, positive-thinking, thinking, transform)",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                     | sentence_1                                                                          | label                                                         |
  |:--------|:-------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                         | string                                                                              | float                                                         |
  | details | <ul><li>min: 3 tokens</li><li>mean: 3.0 tokens</li><li>max: 3 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 54.13 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0         | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | label            |
  |:-------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>quote</code> | <code>“have you ever noticed how â€˜what the hellâ€™ is always the right decision to make?” - terry johnson, (humor, misattributed-to-marilyn-monroe, philosophy)</code>                                                                                                                                                                                                                                                                                                                                                                                                                                             | <code>1.0</code> |
  | <code>quote</code> | <code>“a human being is a part of the whole called by us universe, a part limited in time and space. he experiences himself, his thoughts and feeling as something separated from the rest, a kind of optical delusion of his consciousness. this delusion is a kind of prison for us, restricting us to our personal desires and to affection for a few persons nearest to us. our task must be to free ourselves from this prison by widening our circle of compassion to embrace all living creatures and the whole of nature in its beauty.” - albert einstein (compassion, einstein, nature, philosophy)</code> | <code>1.0</code> |
  | <code>quote</code> | <code>“it's far better to be unhappy alone than unhappy with someone â€” so far.” - marilyn monroe (alone)</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.12.6
- Sentence Transformers: 4.1.0
- Transformers: 4.52.3
- PyTorch: 2.7.0+cpu
- Accelerate: 1.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->