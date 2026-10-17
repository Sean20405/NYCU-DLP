# Create env
conda create -y -n dlp_lab7 python=3.11
conda activate dlp_lab7
pip install -r src/requirements.txt

python src/test_model_task1.py task1_a2c_pendulum.pth

python src/test_model_task2.py task2_ppo_pendulum.pth

python src/test_model_task3.py .
