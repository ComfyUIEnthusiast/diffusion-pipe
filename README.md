# Docker_Diffusion-Pipe

Run this in the diffusion-pipe directory:
	NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /configs/config.toml
	
To continue from checkpoint:
	NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /configs/config.toml --resume_from_checkpoint