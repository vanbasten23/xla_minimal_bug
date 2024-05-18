import torch_xla.distributed.xla_multiprocessing as xmp
import torch.distributed as dist

# Note that we defer the `train` import until *after* we've initialized the XLA context!
def xla_train(_device_id: int) -> None:
    # dist.init_process_group("xla", init_method="xla://")
    # dist.barrier()
    from main import main
    main()


if __name__ == "__main__":
    xmp.spawn(xla_train, args=())
