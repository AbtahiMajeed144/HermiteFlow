python src/video_Nx.py \
  -m configs/hermiteflow/hermiteflow_r.yaml \
  -l path/to/checkpoint.ckpt \
  --eval \
  --source-path path/to/input_frames \
  --output-path path/to/output_dir \
  --N 8 \
  --ds-factor 1.0
