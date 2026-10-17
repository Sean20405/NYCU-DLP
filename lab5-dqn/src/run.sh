# Create env
conda create -y -n dlp_lab5 python=3.12
conda activate dlp_lab5
pip install -r src/requirements.txt

python src/test_model_task1.py \
    --model-path task1.pt \
    --output-dir eval_videos/task1

python src/test_model_task2.py \
    --model-path task2.pt \
    --output-dir eval_videos/task2

python src/test_model_task3.py \
    --model-dir . \
    --output-dir eval_videos/task3
