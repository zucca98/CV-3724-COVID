# Lightweight, CPU-only image to reproduce the COVID-CT classification project.
# Scope: classical pipeline (SVM/RF), tests, and the Streamlit demo. Deep-model
# training (scripts/train_deep.py) is intentionally out of scope here - it
# needs a CUDA GPU and is run either on the host (Option A in the README) or
# on Colab. Keeping this image CPU-only keeps it small and portable for graders.
# Build:   docker build -t covid-ct .
# Run CLI: docker run --rm -v ${PWD}:/app -w /app covid-ct pytest tests/ -v
# Run demo: docker run --rm -p 8501:8501 -v ${PWD}:/app -w /app covid-ct \
#               streamlit run app/demo.py --server.address 0.0.0.0

FROM python:3.12-slim

# System libs needed by OpenCV and lungmask.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (smaller wheels, separate index).
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# Install project dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt streamlit

# Copy the source last so dependency layers stay cached.
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["pytest", "tests/", "-v"]
