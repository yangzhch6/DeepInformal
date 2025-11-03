# Tested with 2 & 4 GPUs

set -x
export WANDB_API_KEY=$yangzhch6_WANDB_API_KEY

nproc_per_node=8

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/workspace/lark/DeepInformal/data/synthetic_align/qwen3-4b-openr1-46k.parquet \
    data.val_files=$HOME/workspace/lark/DeepInformal/data/synthetic_align/qwen3-4b-openr1-46k-val.parquet \
    model.partial_pretrain=$HOME/workspace/lark/models/Qwen/Qwen3-4B \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=1024 \
    data.max_length=36864 \
    optim.lr=1e-6 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.default_local_dir=$HOME/workspace/lark/DeepInformal/checkpoints/synthetic_align/qwen3-4b \
    trainer.project_name=synthetic_align \
    trainer.experiment_name=qwen3-4b-openr1-46k \
    trainer.total_epochs=3 \
    trainer.logger='["console","wandb"]' $@