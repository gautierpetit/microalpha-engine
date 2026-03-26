# Docker
- Build: docker build -t microalpha .
- Test: docker run --rm microalpha pytest -q
- Run: docker run --rm \
  --mount type=bind,source="$(pwd)/data",target=/app/data \
  --mount type=bind,source="$(pwd)/artifacts",target=/app/artifacts \
  microalpha