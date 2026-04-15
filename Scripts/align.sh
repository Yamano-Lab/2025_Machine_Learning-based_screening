python ./src/create_aligned_dataset_v5.py \
    --input_dir /Users/matsuokoujirou/Documents/Data/Screening/Candidates/260205/ \
    --output_dir ./data/aligned_dataset_v5/Candidates_01 \
    --workers 4




python src/visualize_average_error.py \
    --input_dir ./data/aligned_dataset_v5/all2 \
    --model_dir ./results/Models/Aligned_20260115 \
    --output_dir ./results/Average_Anomalies/journal_quality

python ./src/split_wt_for_validation.py \
    --input_dir ./data/aligned_dataset_v5 \
    --ratio 0.2