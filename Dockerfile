FROM python:3.13-slim
WORKDIR /work
COPY requirements.txt .
# CPU-only PyTorch wheels (the default PyPI wheels pull in ~2 GB of CUDA libraries)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY scripts scripts
CMD ["python", "scripts/train_classifier.py"]
