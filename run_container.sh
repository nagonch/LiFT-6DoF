docker run -it --rm \
  --gpus all \
  -v "$(pwd):/LiFT-6DoF" \
  -w /LiFT-6DoF \
  lift6dof \
  bash
