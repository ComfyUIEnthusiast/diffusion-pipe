# Docker_Diffusion-Pipe
Currently I have to pip uninstall flash-attn from the command line and re pip install it like below. Not sure why:

    pip uninstall flash-attn
	FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
	
Add a wslconfig to grant more shared memory to the wsl environment if running on windows:

	[wsl2]
	memory=48GB
	swap=64GB

Update the dataset.toml to point at the correct dataset, and update settings in config.toml

Run this in the diffusion-pipe directory:

	NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /configs/config.toml
	
To continue from checkpoint:

	NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /configs/config.toml --resume_from_checkpoint