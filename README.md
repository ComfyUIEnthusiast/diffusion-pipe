# Docker_Diffusion-Pipe
Currently I have to re-install flash-attn during the startup script. Not sure why, probably because I don't have versions defined:
	
Add a wslconfig to grant more shared memory to the wsl environment if running on windows:

	[wsl2]
	memory=48GB
	swap=64GB

Update the dataset.toml to point at the correct dataset, and update settings in config.toml

Run this in the diffusion-pipe directory:

	NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /configs/config.toml
	
To continue from checkpoint:

	NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /configs/config.toml --resume_from_checkpoint
