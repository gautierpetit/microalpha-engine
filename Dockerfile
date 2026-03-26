FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY microalpha ./microalpha
COPY scripts ./scripts
COPY config ./config
COPY cpp ./cpp
COPY tests ./tests

RUN pip install --upgrade pip
RUN pip install pybind11
RUN pip install .[dev]

# Build the C++ extension with CMake and copy it into /app/microalpha
RUN cmake -S cpp -B cpp/build \
    -DPYBIND11_FINDPYTHON=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
 && cmake --build cpp/build --config Release

CMD ["python", "-m", "scripts.run_experiment"]