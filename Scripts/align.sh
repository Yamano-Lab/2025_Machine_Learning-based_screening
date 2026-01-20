python ./src/create_aligned_dataset_v4.py \
    --input_dir ./data/raw_images \
    --output_dir ./data/aligned_dataset_v4 \
    --workers 4




python src/visualize_average_error.py \
    --input_dir ./data/aligned_dataset_v5/all \
    --model_dir ./results/Models/Aligned_20260115 \
    --output_dir ./results/Average_Anomalies/all