# NYCU Deep Learning & Deep Learning Labs (DLP)

My code, results, and reports for the labs of the Deep Learning course at National Yang Ming Chiao Tung University (NYCU), Spring 2026.

| Lab | Topic | Key result | Folder |
|---|---|---|---|
| 1 | Backpropagation, MLP in NumPy | 100% test accuracy on the linear and XOR datasets | [lab1-backpropagation](lab1-backpropagation) |
| 2 | Binary semantic segmentation (Oxford-IIIT Pet)<br><img src="lab2-segmentation/results/resnet_unet_samples.png" alt="Segmentation samples" width="260"> | Kaggle Dice 0.9341 (ResNet34-UNet), 0.9105 (UNet) | [lab2-segmentation](lab2-segmentation) |
| 3 | MaskGIT image inpainting<br><img src="lab3-maskgit/report/assets/exp-vis.png" alt="MaskGIT inpainting before and after" width="260"> | Best FID 29.14 with the cosine mask schedule | [lab3-maskgit](lab3-maskgit) |
| 4 | Conditional VAE for video prediction<br><img src="lab4-cvae-video/results/gifs/seq0_input_vs_pred.gif" alt="Input and predicted video" width="260"> | Validation PSNR about 39.8 | [lab4-cvae-video](lab4-cvae-video) |
| 5 | DQN, Double DQN, PER, Dueling, multi-step (CartPole, Pong)<br><img src="lab5-dqn/report/assets/task3_curve.png" alt="Pong evaluation score" width="260"> | Pong average reward 19.95 | [lab5-dqn](lab5-dqn) |
| 6 | Conditional diffusion on iCLEVR<br><img src="lab6-diffusion/report/assets/result-test_grid.png" alt="Generated iCLEVR images" width="260"> | Classifier accuracy 0.8611 (test), 0.9048 (new test) | [lab6-diffusion](lab6-diffusion) |
| 7 | A2C and PPO (Pendulum, Walker2d)<br><img src="lab7-a2c-ppo/report/assets/task3_curve.png" alt="Walker2d evaluation return" width="260"> | Walker2d average return up to 4240 | [lab7-a2c-ppo](lab7-a2c-ppo) |

Each lab folder has the same layout.

- `README.md` describes the task, the setup, the run commands, and the main results.
- `src/` holds the submitted code.
- `results/` holds plots, tables, and sample outputs.
- `report/` holds the report as PDF and LaTeX source.

## Academic integrity

Use this repository as a reference. Do not submit it as your own work. Your course probably prohibits plagiarism.

## License

The code is under the MIT License. The reports are shared for reading only.
