Minimal code piece to reproduce our Pytorch XLA error

On a multi-GPU machine:
```
export CUDA_VISIBLE_DEVICES=0,1
PJRT_DEVICE=CUDA GPU_NUM_DEVICES=2 python xla_entry.py
```
The program will fail at line 42 of main `batch = next(iter(dataloader))`

By removing `xm.is_master_ordinal()` at line 7 of main, the program runs fine.
