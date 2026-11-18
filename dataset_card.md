# Animal-MNIST — Dataset Card

## Summary

Animal-MNIST is a compact, class-balanced image-classification benchmark containing **10,000 synthetic animal images** across ten categories. Images are supplied as **28 × 28, single-channel, 8-bit grayscale** arrays, in the spirit of MNIST. Its primary purpose is to support accessible experiments in small-image classification and the study of shape-related visual information.

The images emphasize animal silhouettes and outlines, but they are **not pure silhouette masks**. Grayscale images can retain internal visual cues.

## Dataset contents

| Attribute | Description |
| --- | --- |
| Images | 10,000 |
| Categories | Bear, Bird, Cat, Cow, Dog, Elephant, Giraffe, Horse, Sheep, Zebra |
| Class balance | 1,000 images per category |
| Image dimensions | 28 × 28 pixels |
| Color and type | Single-channel grayscale, `uint8`, intensity range 0–255 |
| Labels | Ten categorical animal-class labels |
| Origin | Synthetic images generated using image-diffusion models, followed by detection-assisted extraction and preprocessing |
| Official evaluation partition | 7,200 training, 800 validation, and 2,000 test images, stratified by class |

The fixed split contains **720 training, 80 validation, and 200 test images per class**. Its indices are distributed in [`data/Animal_MNIST/animal_mnist_splits_v5.npz`](data/Animal_MNIST/animal_mnist_splits_v5.npz). Use these indices when comparing results with experiments that report the official split. Studies using cross-validation or alternative partitions should identify their own evaluation protocol explicitly.

## How the data were created

The dataset was assembled from generated animal imagery, primarily using **Proteus v0.6**, with **Stable Diffusion v1.5** also used. Animal candidates were isolated using Ultralytics YOLO detectors and then cropped, centered, standardized, and reduced to the benchmark format. Candidates underwent visual review and duplicate screening.

The release was assembled over multiple generation and processing batches. Historical per-image random seeds, exact checkpoint revisions, complete prompts, and per-class rejection logs were not retained. Consequently, the documented workflow is reproducible in outline, but **the precise set of released images cannot be regenerated deterministically** from the available records.

For the documented prompts, model settings, applicable library defaults, detection thresholds, preprocessing steps, and limits of reproducibility, see [`GENERATION_PROTOCOL.md`](GENERATION_PROTOCOL.md). The public generator is an implementation example rather than a complete historical record of every image.



## Appropriate uses

- Teaching and demonstration of image classification, data preprocessing, and evaluation.
- Comparisons of classical machine-learning models, convolutional neural networks, and other architectures on compact grayscale inputs.
- Experiments examining the relative usefulness of grayscale, filled-mask, and boundary representations, provided the transformations and training protocols are disclosed.
- Reproducible model comparison with the published split indices and clearly reported random seeds and hyperparameters.

## Limitations and cautions

- **Synthetic source:** Generated animal imagery differs from photographs and field observations. Accuracy here does not establish performance on real-world animal photographs.
- **Limited coverage:** There are only ten categories; several depict four-legged mammals. This is not a comprehensive representation of animal diversity.
- **Shape emphasis, not shape exclusivity:** The low-resolution grayscale format suppresses much color and texture detail, but internal grayscale patterns and other shortcuts may remain. Animal categories with similar outlines, particularly Horse and Zebra, can be difficult to distinguish using shape alone.
- **Selection bias:** Detection-assisted extraction and human review may favor certain poses, shapes, and visually clear candidates.
- **No fairness or deployment validation:** The dataset is not intended to support operational wildlife identification, ecological surveillance, safety-critical decisions, or claims about model performance outside the benchmark.

## Access and citation

- Project repository: [Animal-MNIST on GitHub](https://github.com/SinaSR-6/Animal-MNIST)
- Archived project release: [Zenodo DOI: 10.5281/zenodo.20991252](https://doi.org/10.5281/zenodo.20991252)
- Generation documentation: [`GENERATION_PROTOCOL.md`](GENERATION_PROTOCOL.md)

Please cite the project's associated manuscript when available, together with the archived release used in your work. For reproducibility, report the evaluation partition and any transformations applied to the released images.

## Licensing and third-party materials

The repository includes an **MIT software license**. **A separate, explicit license statement for the released dataset images has not been verified in the available project records.** Do not assume that the software license also determines the terms for every generated image. The maintainers should clarify the dataset-image reuse terms in the repository and archived record before those permissions are represented as settled.

# Animal-MNIST Dataset License

The Animal-MNIST dataset, including its images and corresponding labels, is distributed under the Creative Commons Attribution 4.0 International License (CC BY 4.0), to the extent that the dataset creators hold the necessary rights.

License:
https://creativecommons.org/licenses/by/4.0/

When using or redistributing Animal-MNIST, please
acknowledge the dataset creators:

Sina Mohammadi, Mohammad Samy Baladram,
and Michael R. Zielewski.

Project:
https://github.com/SinaSR-6/Animal-MNIST

This license applies to the Animal-MNIST dataset. It does not relicense the third-party generative models, object-detection models, or other dependencies used during its creation.

The pretrained models used in dataset production are third-party artifacts, whose terms are **independent of the repository's MIT software license**:

- **Stable Diffusion v1.5:** The [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) identifies the **CreativeML OpenRAIL-M** license.
- **Proteus v0.6:** The [model card](https://huggingface.co/dataautogpt3/Proteus-v0.6) discusses **PolyForm Small Business License 1.0.0 as a contemplated option**; this discussion should not be treated as confirmation of a final license grant. Verify applicable terms with the model provider.
- **Ultralytics YOLOv8:** The [project documentation](https://docs.ultralytics.com/models/yolov8/) identifies **AGPL-3.0 and enterprise licensing** for its software and models.

Consult the applicable third-party terms before reusing these model weights or generating additional images. Distribution of Animal-MNIST does not relicense upstream models.
