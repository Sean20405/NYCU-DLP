python dqn_task3.py \
    --exp-name all \
    --dueling --double-q --per --multi-step 3 --action-reduction

python dqn_task3.py \
    --exp-name all_noActReduce \
    --dueling --double-q --per --multi-step 3 

python dqn_task3.py \
    --exp-name all_noMultiStep \
    --dueling --double-q --per --action-reduction

python dqn_task3.py \
    --exp-name all_noPER \
    --dueling --double-q --multi-step 3  --action-reduction

python dqn_task3.py \
    --exp-name all_noDoubleQ \
    --dueling --per --multi-step 3  --action-reduction

python dqn_task3.py \
    --exp-name all_noDuel \
    --double-q --per --multi-step 3  --action-reduction