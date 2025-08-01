# add --wfep if you want use the Waveform Data Enhancement
torchrun --nproc_per_node 8 -m training.main --save-frequency 1 --val-frequency 5 --report-to tensorboard --mimic-iv-ecg-path="..\data\mimic-iv\"  --warmup 10000  --batch-size 96 --lr 1e-4 --wd 0.1 --epochs 20 --workers 8 --model coca_ViT-B-32 --lock-text --grad-clip-norm 0.5


# --ptbxl-path="/path/to/ptb-xl" --cpsc2018-path="/path/to/icbeb" --champan-path="/path/to/Champan-Shaoxing" --sph-path="/path/to/SPH" --lock-text --grad-clip-norm 0.5


