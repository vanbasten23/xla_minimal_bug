Minimal code piece to reproduce our Pytorch XLA error

```
export CUDA_VISIBLE_DEVICES=1,2
PJRT_DEVICE=CUDA GPU_NUM_DEVICES=2 python xla_entry.py
```