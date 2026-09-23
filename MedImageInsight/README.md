# MedImageInsight

An implementation of **MedImageInsight**, an open-source embedding model for general-domain medical imaging. The package supports zero-shot classification, image-image search, embedding extraction, preprocessing for common medical image formats, and ROC curve generation.

This project is based on the MedImageInsight paper:
<https://arxiv.org/abs/2410.06542>

## Research Use Only

This software is intended for research and educational use. It is not a medical device and must not be used for clinical diagnosis, treatment decisions, or other patient-care purposes.

## Requirements

- Python 3.9 or newer
- A compatible PyTorch installation
- Access to the model weights used by the implementation

## Installation

Clone the repository and install the package in a virtual environment:

```bash
git clone https://github.com/YOUR_USERNAME/MedImageInsight.git
cd MedImageInsight

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Examples

Run zero-shot classification with an image:

```bash
python examples/01_zero_shot_classification.py --image path/to/image.png
```

Choose a supported imaging domain:

```bash
python examples/01_zero_shot_classification.py \
  --image path/to/image.png \
  --domain chest_xray
```

Other examples are available in the `examples/` directory:

- `02_image_image_search.py`: find similar reference images
- `03_embedding_extraction.py`: extract image embeddings
- `04_roc_curve_generation.py`: generate ROC curves
- `05_fine_tuning.py`: fine-tune the model for a custom task

Running an example without an image uses a dummy image where supported; model loading may download weights and require network access.

## Package API

The main public classes are:

```python
from medimageinsight import (
    EmbeddingExtractor,
    KNNClassifier,
    MedicalImagePreprocessor,
    MedImageInsightModel,
    ZeroShotClassifier,
)
```

See the example scripts and the `docs/` directory for more detailed workflows.

## Testing

Install the test dependency if needed, then run:

```bash
pip install pytest
pytest
```

## Project Structure

```text
medimageinsight/   Core package
examples/          End-to-end usage examples
docs/              Documentation
tests/             Automated tests
assets/            Project assets
```

## License

See [LICENSE](LICENSE) for licensing information.