# Third-party notices

This repository's original source code is distributed under the MIT License.
Third-party packages, pretrained weights, datasets, and papers remain subject
to their own terms.

## Runtime and training dependencies

Release-container packages are pinned in `requirements-inference.txt`; the
CUDA/PyTorch base is pinned by image digest in `Dockerfile`. Training and
development packages are pinned separately. Their upstream licenses apply
independently.

## Pretrained model

The ConvNeXt-Tiny backbone is loaded through timm as
`convnext_tiny.fb_in22k_ft_in1k` and was pretrained on ImageNet-22k then
fine-tuned on ImageNet-1k. The deploy checkpoint contains the resulting
fine-tuned parameters and does not download weights at runtime.

- timm project: https://github.com/huggingface/pytorch-image-models
- model card: https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k
- ConvNeXt paper: https://arxiv.org/abs/2201.03545
- ImageNet: https://www.image-net.org/

timm source code is Apache-2.0 licensed. Dataset and pretrained-weight terms
may impose additional restrictions; this notice does not grant rights to the
ImageNet data or supersede the upstream model card.

## DINOv2 backbone

The adapted `dinov2_vitb14_reg` branch originates from Meta AI's public
DINOv2 repository. Reproducible training code pins source commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`; release inference uses the
self-contained adapted TorchScript artifact and performs no download.

- Source: https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8
- Model entry point: `dinov2_vitb14_reg`
- Bundled artifact: `weights/release/dinov2_vehicle_cls_tokens.ts`
- SHA-256: `d11e114a429400227f4a5bc374800ac1839076eee807b02db08e251c9d6691e0`
- Project/model terms: see the license files in the pinned upstream source.

## Adapted OSNet branch

The release OSNet branch descends from Open Model Zoo
`vehicle-reid-0001`, then is adapted only on the supplied hackathon train
split. Its self-contained artifact is
`weights/release/osnet_loss_branch_global_parts.ts`; SHA-256:
`666105da046c57e9f1a01fb0058bcf588a07769374797904409397821bf80fe2`.

## Organizer dataset

The hackathon images and annotation CSVs are not distributed by this GitHub
repository. They are mounted locally under `dataset/`. Reproducible split CSVs
and error montages are also excluded from the public release until the data
owner explicitly confirms redistribution rights.

The retained deploy checkpoint uses only the supplied hackathon train data in
addition to the documented public pretrained weights. Closed, proprietary, or
non-reproducible datasets are not used.

## Source vehicle-ReID expert

The adapted release OSNet originates from Open Model Zoo
`vehicle-reid-0001` (`osnet_ain_x1_0_vehicle_reid.onnx`). It is a compact
OSNet-AIN model trained for vehicle re-identification. The original model is
distributed under the MIT License; the Open Model Zoo metadata is Apache-2.0
licensed.

- Model documentation: https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/vehicle-reid-0001
- Official model file: https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx
- Original-model license: https://raw.githubusercontent.com/sovrasov/deep-person-reid/vehicle_reid/LICENSE
- SHA-384: `0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7`

ONNX is used to read the source checkpoint during adaptation and export. The
release runtime uses the self-contained TorchScript artifact.
