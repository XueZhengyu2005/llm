---
license: apache-2.0
task_categories:
- text-generation
language:
- en
- zh
tags:
- dpo
- orpo
- rlhf
- llama-factory
size_categories:
- 10K<n<100K
configs:
- config_name: en
  data_files:
  - split: train
    path: dpo_en.json
- config_name: zh
  data_files:
  - split: train
    path: dpo_zh.json
dataset_info:
- config_name: en
  features:
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: value
      dtype: string
  - name: chosen
    struct:
    - name: from
      dtype: string
    - name: value
      dtype: string
  - name: rejected
    struct:
    - name: from
      dtype: string
    - name: value
      dtype: string
- config_name: zh
  features:
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: value
      dtype: string
  - name: chosen
    struct:
    - name: from
      dtype: string
    - name: value
      dtype: string
  - name: rejected
    struct:
    - name: from
      dtype: string
    - name: value
      dtype: string
---

This dataset is composed by

- 4,000 examples of [argilla/distilabel-capybara-dpo-7k-binarized](https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized) with chosen score>=4.
- 3,000 examples of [argilla/distilabel-intel-orca-dpo-pairs](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) with chosen score>=8.
- 3,000 examples of [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) with chosen score>=4.
- 10,000 examples of [wenbopan/Chinese-dpo-pairs](https://huggingface.co/datasets/wenbopan/Chinese-dpo-pairs).

You can use it in [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) by specifying `dataset: dpo_mix_en,dpo_mix_zh`.
