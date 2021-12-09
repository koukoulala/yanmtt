# 12.9
nohup bash scripts/generate_pretrain_data.sh ./xprophetnet_models ../datasets/small_NTG 2 huggingface_prophetnet_ntg_ckpt_4.pt 1 generate_data &> logs/try_generate.out &
"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_generate_data && . /tmp/env_generate_data/bin/activate && python -m pip install -r requirements.txt && bash scripts/generate_pretrain_data.sh [#input-previous-model-path] [#input-training-data-path] 24 huggingface_prophetnet_sampled_aug_filter.pt 1 generate_data"
"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_full_generate_data && . /tmp/env_full_generate_data/bin/activate && python -m pip install -r requirements.txt && bash scripts/generate_pretrain_data.sh [#input-previous-model-path] [#input-training-data-path] 24 huggingface_prophetnet_sampled_aug_filter.pt 1 generate_data"

export CUDA_VISIBLE_DEVICES=0,1
python pretrain_mbart.py -n 1  -nr 0 -g 2 --model_path examples/models/mbart_model --tokenizer_name_or_path examples/tokenizers/albert-vienhi16k --langs hi,en,vi --mono_src examples/data/train.hi,examples/data/train.en,examples/data/train.vi --encoder_layers 1 --decoder_layers 1 --encoder_attention_heads=1 --decoder_attention_heads=1 --encoder_ffn_dim=128 --decoder_ffn_dim=128 --d_model=64 --shard_files

