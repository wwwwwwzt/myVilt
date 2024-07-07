---
license: apache-2.0
tags:
- vision
- image-classification
datasets:
- imagenet-1k
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace
---

# Swin Transformer v2 (tiny-sized model) 

Swin Transformer v2 model pre-trained on ImageNet-1k at resolution 256x256. It was introduced in the paper [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883) by Liu et al. and first released in [this repository](https://github.com/microsoft/Swin-Transformer). 

Disclaimer: The team releasing Swin Transformer v2 did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

The Swin Transformer is a type of Vision Transformer. It builds hierarchical feature maps by merging image patches (shown in gray) in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window (shown in red). It can thus serve as a general-purpose backbone for both image classification and dense recognition tasks. In contrast, previous vision Transformers produce feature maps of a single low resolution and have quadratic computation complexity to input image size due to computation of self-attention globally.

Swin Transformer v2 adds 3 main improvements: 1) a residual-post-norm method combined with cosine attention to improve training stability; 2) a log-spaced continuous position bias method to effectively transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs; 3) a self-supervised pre-training method, SimMIM, to reduce the needs of vast labeled images.

![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/swin_transformer_architecture.png)

[Source](https://paperswithcode.com/method/swin-transformer)

## Intended uses & limitations

You can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=swinv2) to look for
fine-tuned versions on a task that interests you.

### How to use

Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

For more code examples, we refer to the [documentation](https://huggingface.co/transformers/model_doc/swinv2.html#).

### BibTeX entry and citation info

```bibtex
@article{DBLP:journals/corr/abs-2111-09883,
  author    = {Ze Liu and
               Han Hu and
               Yutong Lin and
               Zhuliang Yao and
               Zhenda Xie and
               Yixuan Wei and
               Jia Ning and
               Yue Cao and
               Zheng Zhang and
               Li Dong and
               Furu Wei and
               Baining Guo},
  title     = {Swin Transformer {V2:} Scaling Up Capacity and Resolution},
  journal   = {CoRR},
  volume    = {abs/2111.09883},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.09883},
  eprinttype = {arXiv},
  eprint    = {2111.09883},
  timestamp = {Thu, 02 Dec 2021 15:54:22 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-09883.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```