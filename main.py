from torchvision import datasets
from torchvision.transforms import ToTensor
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl


xm.is_master_ordinal() # remove this line will make the code work but we really need it for certain reason in our actual codebase

def naive_worker_init_fn(worker_id: int) -> None:
    pass

def main() -> None:
    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    collator = None

    from torch.utils.data import DataLoader, DistributedSampler

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
        seed=0,
        drop_last=False,
    )
    worker_init_fn = naive_worker_init_fn
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collator,
        num_workers=2,
        # worker_init_fn=worker_init_fn,
    )
    dataloader = pl.MpDeviceLoader(dataloader, xm.xla_device())
    print("healthy before batch")
    for data, target in dataloader:
        pass
    print("healthy after batch")
