# -*- coding: utf-8 -*-
# Copyright 2021 National Institute of Information and Communication Technology (Raj Dabre)
# 
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the
# Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall
# be included in all copies or substantial portions of the
# Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Basic imports
import os
import sys
import argparse
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
##

## Huggingface imports
import transformers
from transformers import AutoTokenizer, MBartTokenizer, MBart50Tokenizer, BartTokenizer
from transformers import MBartForConditionalGeneration, BartForConditionalGeneration, MBartConfig, BartConfig, get_linear_schedule_with_warmup
from transformers import AdamW
##

## Pytorch imports
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
##

## Our imports
from common_utils import *
##

## Other imports
import math
import random
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
import gc
import functools
from prefetch_generator import BackgroundGenerator
##

## Seed setting here
torch.manual_seed(621311)
##

def model_create_load_run_save(gpu, args, train_files, dev_files):
    """The main function which does the overall training. Should be split into multiple parts in the future. Currently monolithc intentionally."""
    
    rank = 0 ## The rank of the current process out of the total number of processes indicated by world_size.
    print("Launching process:", rank)

    if args.shard_files: ## First shard the data using process 0 aka the prime process or master process. Other processes will wait.
        shard_files_bi(train_files, args)

    tok = MBartTokenizer.from_pretrained(args.tokenizer_name_or_path)
    print("Tokenizer is:", tok)
    
    print("We won't do fp32 training")
    
    if args.encoder_tying_config is not None:
        print("We will use recurrently stacked layers for the encoder with configuration:", args.encoder_tying_config)
    if args.decoder_tying_config is not None:
        print("We will use recurrently stacked layers for the decoder with configuration:", args.decoder_tying_config)
    
    if args.unidirectional_encoder:
        print("Using unidirectional encoder.")
    
    if args.use_official_pretrained:
        config = MBartConfig.from_pretrained(args.pretrained_model)
        model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model, config=config) ## We may use FBs official model and fine-tune it for our purposes.
    else:
        config = MBartConfig(vocab_size=len(tok), encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, dropout=args.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, encoder_attention_heads=args.encoder_attention_heads, decoder_attention_heads=args.decoder_attention_heads, encoder_ffn_dim=args.encoder_ffn_dim, decoder_ffn_dim=args.decoder_ffn_dim, d_model=args.d_model, no_embed_norm=args.no_embed_norm, scale_embedding=args.scale_embedding, pad_token_id=tok.pad_token_id, eos_token_id=tok(["</s>"], add_special_tokens=False).input_ids[0][0], bos_token_id=tok(["<s>"], add_special_tokens=False).input_ids[0][0], encoder_tying_config=args.encoder_tying_config, decoder_tying_config=args.decoder_tying_config, multilayer_softmaxing=args.multilayer_softmaxing, wait_k=args.wait_k, additional_source_wait_k=args.additional_source_wait_k, unidirectional_encoder=args.unidirectional_encoder, multi_source=args.multi_source, multi_source_method=args.multi_source_method, softmax_temperature=args.softmax_temperature, temperature_calibration=args.temperature_calibration, encoder_layerdrop=args.layerdrop, decoder_layerdrop=args.layerdrop, no_scale_attention_embedding=args.no_scale_attention_embedding, positional_encodings=args.positional_encodings, num_domains_for_domain_classifier=args.num_domains_for_domain_classifier, gradient_reversal_for_domain_classifier=args.gradient_reversal_for_domain_classifier) ## Configuration. TODO: Save this configuration somehow.
        model = MBartForConditionalGeneration(config)

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("n_gpu=", n_gpu, device)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print("Using label smoothing of", args.label_smoothing)
    print("Using gradient clipping norm of", args.max_gradient_clip_value)
    print("Using softmax temperature of", args.softmax_temperature)

    individual_sbleu_history = {dev_pair: [] for dev_pair in dev_files} ## For multilingual NMT settings we suppose that we will keep a track of the histories for individual language pairs being evaluated and this dictionary keeps track of the history.

    inps = {dev_pair: [inpline.strip() for inpline in open(dev_files[dev_pair][0])][:args.max_eval_batches*args.dev_batch_size] for dev_pair in dev_files} ## Get all inputs for each pair. Select up to args.max_eval_batches*args.dev_batch_size examples.
    refs = {dev_pair: [[refline.strip() for refline in open(dev_files[dev_pair][1])][:args.max_eval_batches*args.dev_batch_size]] for dev_pair in dev_files} ## Get all references for each input. Select up to args.max_eval_batches*args.dev_batch_size examples.
    scores = {dev_pair: 0 for dev_pair in dev_files} ## The rouge scorer works at the sentence level so we have to add all individual scores per sentence and this dictionary keeps track of the score. This dictionary may not be needed.

    start = time.time()
    hyp = {dev_pair: [] for dev_pair in dev_files}
    sbleus = {}
    model.eval()  ## We go to eval mode so that there will be no dropout.

    for dev_pair in dev_files:  ## For each evaluation pair we will decode and compute scores.
        print("dev_pair", dev_pair)
        slangtlang = dev_pair.strip().split("-")
        slang = slangtlang[0]
        tlang = slangtlang[1]
        for dev_input_ids, dev_input_masks in generate_batches_eval_bilingual(tok, args, inps[dev_pair], slang):
            start = time.time()
            dev_input_ids = dev_input_ids.to(device)  ## Move to GPU.
            dev_input_masks = dev_input_masks.to(device)  ## Move to GPU.
            print("Decoding batch from a pool of", len(inps[dev_pair]), "examples")
            with torch.no_grad():  ## torch.no_grad is apparently known to prevent the code from allocating memory for gradient computation in addition to making things faster. I have not verified this but have kept it as a safety measure to ensure that my model is not being directly tuned on the development set.
                model_to_generate = (
                    model.module if hasattr(model, "module") else model
                )
                translations = model_to_generate.generate(dev_input_ids, use_cache=True, num_beams=1,
                                                     max_length=int((len(dev_input_ids[0]) * args.max_decode_length_multiplier) if args.max_decode_length_multiplier > 0 else -args.max_decode_length_multiplier),
                                                     min_length=int((len(dev_input_ids[0]) * args.min_decode_length_multiplier) if args.min_decode_length_multiplier > 0 else -args.min_decode_length_multiplier),
                                                     early_stopping=True, attention_mask=dev_input_masks,
                                                     pad_token_id=tok.pad_token_id,
                                                     eos_token_id=tok(["</s>"], add_special_tokens=False).input_ids[0][0],
                                                     decoder_start_token_id=tok(tlang, add_special_tokens=False).input_ids[0][0],
                                                     bos_token_id=tok(["<s>"], add_special_tokens=False).input_ids[0][0],
                                                     length_penalty=args.length_penalty,
                                                     repetition_penalty=args.repetition_penalty,
                                                     encoder_no_repeat_ngram_size=args.encoder_no_repeat_ngram_size,
                                                     no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                     additional_input_ids=None,
                                                     additional_input_ids_mask=None)  ## We translate the batch.
            translations = translations.to('cpu')  ## Move to cpu. Not needed but its a safe step.
            for translation in translations:
                translation = tok.decode(translation, skip_special_tokens=args.no_skip_special_tokens,
                                         clean_up_tokenization_spaces=False)  ### Get the raw sentences.
                hyp[dev_pair].append(translation)

        sbleu = get_sacrebleu(refs[dev_pair], hyp[dev_pair])
        metric = 'BLEU'

        individual_sbleu_history[dev_pair].append(sbleu)  ## Update the score history for this pair.
        sbleus[dev_pair] = sbleu
        print(metric, "score using sacrebleu is", sbleu, "for language pair", dev_pair)

        # output to generation file
        with open(os.path.join(args.model_path, dev_pair + ".txt"), 'w') as f:
            for line in hyp[dev_pair]:
                f.write(line + '\n')
    

def run_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-a', '--ipaddr', default='localhost', type=str, 
                        help='IP address of the main node')
    parser.add_argument('-p', '--port', default='26023', type=str, 
                        help='Port main node')
    parser.add_argument('--freeze_embeddings', action='store_true', 
                        help='Should freeze embeddings during fine tuning?')
    parser.add_argument('--freeze_encoder', action='store_true', 
                        help='Should we freeze encoder during fine tuning?')
    parser.add_argument('--positional_encodings', action='store_true', 
                        help='If true then we will use positional encodings instead of learned positional embeddings.')
    parser.add_argument('--no_embed_norm', action='store_true', 
                        help='If true then we wont normalize embeddings.')
    parser.add_argument('--scale_embedding', action='store_true', 
                        help='Should we scale embeddings?')
    parser.add_argument('--no_scale_attention_embedding', action='store_true', 
                        help='Should we scale attention embeddings?')
    parser.add_argument('--multistep_optimizer_steps', default=1, type=int, help="In case you want to simulate a larger batch you should set this to a higher value.")
    parser.add_argument('--encoder_layers', default=6, type=int, help="The value for number of encoder layers")
    parser.add_argument('--decoder_layers', default=6, type=int, help="The value for number of decoder layers")
    parser.add_argument('--label_smoothing', default=0.1, type=float, help="The value for label smoothing")
    parser.add_argument('--weight_decay', default=0.0001, type=float, help="The value for weight decay")
    parser.add_argument('--lr', default=7e-4, type=float, help="The value for the learning rate")
    parser.add_argument('--layerdrop', default=0.0, type=float, help="The value for layerdrop which indicates the probability that a whole layer will be bypassed via an identity transformation.")
    parser.add_argument('--dropout', default=0.1, type=float, help="The value for embedding dropout")
    parser.add_argument('--attention_dropout', default=0.1, type=float, help="The value for attention dropout")
    parser.add_argument('--activation_dropout', default=0.1, type=float, help="The value for activation dropout")
    parser.add_argument('--data_sampling_temperature', default=5.0, type=float, help="The value for the data sampling temperature")
    parser.add_argument('--token_masking_lambda', default=3.5, type=float, help="The value for the poisson sampling lambda value")
    parser.add_argument('--token_masking_probs_range', nargs='+', type=float, default=[0.3], help="The range of probabilities with which the token will be masked. If you want a fixed probability then specify one argument else specify ONLY 2.")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, 
                        help='To prevent repetition during decoding. 1.0 means no repetition. 1.2 was supposed to be a good value for some settings according to some researchers.')
    parser.add_argument('--no_repeat_ngram_size', default=0, type=int, 
                        help='N-grams of this size will never be repeated in the decoder. Lets play with 2-grams as default.')
    parser.add_argument('--length_penalty', default=1.0, type=float, 
                        help='Set to more than 1.0 for longer sentences.')
    parser.add_argument('--no_skip_special_tokens', action='store_false', 
                        help='Should we return outputs without special tokens? We may need this to deal with situations where the user specified control tokens must be in the output.')
    parser.add_argument('--encoder_no_repeat_ngram_size', default=0, type=int, 
                        help='N-gram sizes to be prevented from being copied over from encoder. Lets play with 2-grams as default.')
    parser.add_argument('--encoder_tying_config', default=None, type=str, 
                        help='What should be the parameter tying configuration? 1-1-1-1-1-1 means 6 layers where all are shared. 1-1-2-2-3-3 means 6 layers, 3 unique layers and each one is recurred twice before passing to another layer. 1-2-3-1-2-3 means 6 layers, 3 unique layers and recurrence is done twice after all layers have been passed through. The default None implies a 1-2-3-4-...-N setup')
    parser.add_argument('--decoder_tying_config', default=None, type=str,
                        help='What should be the parameter tying configuration? 1-1-1-1-1-1 means 6 layers where all are shared. 1-1-2-2-3-3 means 6 layers, 3 unique layers and each one is recurred twice before passing to another layer. 1-2-3-1-2-3 means 6 layers, 3 unique layers and recurrence is done twice after all layers have been passed through. The default None implies a 1-2-3-4-...-N setup')
    parser.add_argument('--softmax_temperature', default=1.0, type=float, help="The value for the softmax temperature")
    parser.add_argument('--distillation_temperature', default=1.0, type=float, help="The value for the softmax temperature during distillation")
    parser.add_argument('--temperature_calibration', action='store_true', 
                        help='Are we calibrating the temperature automatically during training? If yes then the softmax_temperature parameter should have a value of 1.0 furthermore the returned temperature will be used to scale the loss.')
    parser.add_argument('--encoder_attention_heads', default=8, type=int, help="The value for number of encoder attention heads")
    parser.add_argument('--decoder_attention_heads', default=8, type=int, help="The value for number of decoder attention heads")
    parser.add_argument('--wait_k', default=-1, type=int, help="The value for k in wait-k snmt. Keep as -1 for non-snmt aka vanilla NMT.")
    parser.add_argument('--mixed_wait_k', action='store_true', 
                        help='Should we train using up to wait_k? This can help simulate multiple wait_k')
    parser.add_argument('--additional_source_wait_k', default=-1, type=int, help="The value for k in wait-k snmt. Keep as -1 for non-snmt aka vanilla NMT. This is the wait-k for the additional source language. Can be used for simultaneous mutlisource NMT.")
    parser.add_argument('--future_prediction', action='store_true', 
                        help='This assumes that we dont mask token sequences randomly but only after the latter half of the sentence. We do this to make the model more robust towards missing future information. Granted we can achieve this using wait-k but methinks this may be a better way of training.')
    parser.add_argument('--unidirectional_encoder', action='store_true', 
                        help='This assumes that we use a unidirectional encoder. This is simulated via a lower-triangular matrix mask in the encoder. Easy peasy lemon squeazy.')
    parser.add_argument('--decoder_ffn_dim', default=2048, type=int, help="The value for decoder ff hidden dim")
    parser.add_argument('--encoder_ffn_dim', default=2048, type=int, help="The value for encoder ff hidden dim")
    parser.add_argument('--d_model', default=512, type=int, help="The value for model hidden size")
    parser.add_argument('--eval_every', default=1000, type=int, help="The number of iterations after which an evaluation must be done. Also saves a checkpoint every these number of steps.")
    parser.add_argument('--no_eval_save_every', default=10000, type=int, help="The number of iterations after which a model must be force saved in case evaluation is not done.")
    parser.add_argument('--max_gradient_clip_value', default=1.0, type=float, help="The max value for gradient norm value")
    parser.add_argument('--use_official_pretrained', action='store_true', 
                        help='Use this flag if you want the argument "pretrained_model" to specify a pretrained model created by someone else.')
    parser.add_argument('--pretrained_model', default='', type=str, 
                        help='Path to the pretrained model.')
    parser.add_argument('--no_reload_optimizer_ctr_and_scheduler', action='store_true',
                        help='Should we reload the optimizer, counter and secheduler? By default we always reload these. Set this to False if we only want to reload the model params and optimize from scratch.')
    parser.add_argument('-m', '--model_path', default='pytorch.bin', type=str, 
                        help='Path to save the fine tuned model')
    parser.add_argument('--save_intermediate_checkpoints', action='store_true', 
                        help='Use this flag if you want intermediate best checkpoints to be saved. If so then numbers will be attached to the checkpoints.')
    parser.add_argument('--warmup_steps', default=16000, type=int,
                        help='Scheduler warmup steps')
    parser.add_argument('--batch_size', default=2048, type=int, 
                        help='Train batch sizes in tokens')
    parser.add_argument('--batch_size_indicates_lines', action='store_true', 
                        help='Should we batch as a fixed number of lines?')
    parser.add_argument('--dev_batch_size', default=1024, type=int, 
                        help='Dev batch sizes in lines')
    parser.add_argument('--max_src_length', default=512, type=int,
                        help='Maximum token length for source language')
    parser.add_argument('--max_tgt_length', default=512, type=int,
                        help='Maximum token length for target language')
    parser.add_argument('--early_stop_checkpoints', default=10, type=int, 
                        help='Number of checkpoints to wait to see if BLEU increases.')
    parser.add_argument('--learning_rate_scaling', default=2, type=int, 
                        help='How much should the LR be divided by during annealing?. Set num_batches to a larger value or else you will see lr go to zero too soon.')
    parser.add_argument('--max_annealing_attempts', default=2, type=int, 
                        help='Number of times LR should be annealed.')
    parser.add_argument('--additional_early_stop_checkpoints_per_anneal_step', default=5, type=int, 
                        help='How many additional checkpoints should we wait till declaring convergence? This will be multiplied with the annealing step number.')
    parser.add_argument('--num_batches', default=500000, type=int, 
                        help='Number of batches to train on')
    parser.add_argument('--max_eval_batches', default=1000, type=int, 
                        help='These many evaluation batches will be considered. Use a small value like 5 to cover a portion of the evaluation data.')
    parser.add_argument('--max_decode_length_multiplier', default=2.0, type=float, 
                        help='This multiplied by the source sentence length will be the maximum decoding length. If you want to directly specify a particular value then set this to the negative of that value.')
    parser.add_argument('--min_decode_length_multiplier', default=0.1, type=float, 
                        help='This multiplied by the source sentence length will be the minimum decoding length. If you want to directly specify a particular value then set this to the negative of that value.')
    parser.add_argument('--tokenizer_name_or_path', default='ai4bharat/indic-bert', type=str, 
                        help='Name of or path to the tokenizer')
    parser.add_argument('--pretrained_tokenizer_name_or_path', default=None, type=str, 
                        help='Name of or path to the tokenizer of the pretrained model if its different from the current model. This tokenizer will be used for remapping embeddings so as to reuse as many pretrained embeddings as possible.')
    parser.add_argument('--multi_source_method', default=None, type=str, 
                        help='How to merge representations from multiple sources? Should be one of self_relevance_and_merge_after_attention, self_relevance_and_merge_before_attention, merge_after_attention, merge_before_attention. We also need to implement averaging methods such as early averaging (average encoder representations) and late averaging (average softmaxes). Relevance mechanisms should have a separate flag in the future.')
    parser.add_argument('--tokenization_sampling', action='store_true', 
                        help='Should we use stoachastic tokenization aka BPE dropout or Subword regularization?')
    parser.add_argument('--tokenization_nbest_list_size', type=int, default=64, 
                        help='The size of the nbest list when doing stochastic tokenization.')
    parser.add_argument('--tokenization_alpha_or_dropout', type=float, default=0.1, 
                        help='The value of sentence piece regularization amount controlled via alpha or the amount of BPE dropout controlled by dropout.')
    parser.add_argument('--train_slang', default='en', type=str, 
                        help='Source language(s) for training. If you want to specify the domain of the language pair then specify it as language-domain (hyphen in the middle) and make sure to set --num_domains_for_domain_classifier to a value > 1. If you want to specify an additional source then you need to do the same thing but note that you can do multi-source domain classification as its just too much.')
    parser.add_argument('--train_tlang', default='hi', type=str, 
                        help='Target language(s) for training')
    parser.add_argument('--train_src', default='', type=str, 
                        help='Source language training sentences')
    parser.add_argument('--train_tgt', default='', type=str, 
                        help='Target language training sentences')
    parser.add_argument('--dev_slang', default='en', type=str, 
                        help='Source language(s) for training')
    parser.add_argument('--dev_tlang', default='hi', type=str, 
                        help='Target language(s) for training')
    parser.add_argument('--dev_src', default='', type=str, 
                        help='Source language(s) development sentences')
    parser.add_argument('--dev_tgt', default='', type=str, 
                        help='Target language(s) development sentences')
    parser.add_argument('--fp16', action='store_true', 
                        help='Should we use fp16 training?')
    parser.add_argument('--no_eval', action='store_true', 
                        help='Should we skip evaluation?')
    parser.add_argument('--source_masking_for_bilingual', action='store_true', 
                        help='Should we use masking on source sentences when training on parallel corpora?')
    parser.add_argument('--is_summarization', action='store_true', 
                        help='Should we use masking on source sentences when training on parallel corpora?')
    parser.add_argument('--hard_truncate_length', default=0, type=int, 
                        help='Should we perform a hard truncation of the batch? This will be needed to eliminate cuda caching errors for when sequence lengths exceed a particular limit. This means self attention matrices will be massive and I used to get errors. Choose this value empirically.')
    parser.add_argument('--use_rouge', action='store_true', 
                        help='Should we use ROUGE for evaluation?')
    parser.add_argument('--max_ent_weight', type=float, default=-1.0, 
                        help='Should we maximize softmax entropy? If the value is anything between 0 and 1 then yes. If its -1.0 then no maximization will be done.')
    parser.add_argument('--num_domains_for_domain_classifier', type=int, default=1, 
                        help='If we have multiple domains then we should set this to a value higher than one.')
    parser.add_argument('--gradient_reversal_for_domain_classifier', action='store_true', 
                        help='Should we do gradient reversal for the domain classifier? If true then all gradients below the softmax layer (meaning linear projection plus softmax activation) for the classifier will be reversed. Essentially, the representations for two domains will be forced to become more similar. This may in turn be used for style transfer.')
    parser.add_argument('--domain_classifier_loss_weight', type=float, default=0.1, 
                        help='What weight should we give to the domain classifier? 1 minus this weight will be given to the main loss.')
    parser.add_argument('--shard_files', action='store_true', 
                        help='Should we shard the training data? Set to true only if the data is not already pre-sharded.')
    parser.add_argument('--multi_source', action='store_true', 
                        help='Are we doing multisource NMT? In that case you should specify the train_src as a hyphen separated pair indicating the parent language and the child language. You should also ensure that the source file is a tab separated file where each line contains "the parent pair source sentence[tab]child pair source sentence".')
    parser.add_argument('--multilayer_softmaxing', default=None, 
                        help='Should we apply a softmax for each decoder layer? Unsupported for distillation. Only for vanilla training. You have to specify a comma separated list of the intermediate layers which you want to softmax. These go from 0 for the embedding layer to L-2 for the penultimate layer.')
    parser.add_argument('--remap_encoder', default='', type=str, 
                        help='This indicates the remappings for the layer. Example: 1-2,2-4,3-6. The plan is to use these remappings to cut down the model prior to decoding or training. Suppose we have a 6 layer model but we only want to utilize the 2nd, 4th and 6th layer then we will copy the content of the 2nd, 4th and 6th layers to the 1st, 2nd and 3rd layer and delete the former layers from the parameter dictionary. This counts as layer pruning. IMPORTANT NOTE: Ensure that you specify ALL child layer indices you wish mapped. For example if you want 1-2,2-1,3-3 you MUST NOT skip the 3-3 part else it will be deleted from the model dictionary and will be randomly initialized. The loading mechanism is not strict so it will ignore missing or non matching keys. ADDITIONAL NOTE: Load a checkpoint with only the model and not the optimizer to prevent failure as we are not sure if remapping optimizers and learning rate schedulers make sense or not.')
    parser.add_argument('--remap_decoder', default='', type=str, 
                        help='This indicates the remappings for the layer. Example: 1-2,2-4,3-6. The plan is to use these remappings to cut down the model prior to decoding or training. Suppose we have a 6 layer model but we only want to utilize the 2nd, 4th and 6th layer then we will copy the content of the 2nd, 4th and 6th layers to the 1st, 2nd and 3rd layer and delete the former layers from the parameter dictionary. This counts as layer pruning. IMPORTANT NOTE: Ensure that you specify ALL child layer indices you wish mapped. For example if you want 1-2,2-1,3-3 you MUST NOT skip the 3-3 part else it will be deleted from the model dictionary and will be randomly initialized. The loading mechanism is not strict so it will ignore missing or non matching keys. ADDITIONAL NOTE: Load a checkpoint with only the model and not the optimizer to prevent failure as we are not sure if remapping optimizers and learning rate schedulers make sense or not.')
    parser.add_argument('--eliminate_encoder_before_initialization', action='store_true', 
                        help='Lets wipe out the encoder params from the pretrained model before we use it to initialize the current model. This means we have random encoder initialization.')
    parser.add_argument('--eliminate_decoder_before_initialization', action='store_true', 
                        help='Lets wipe out the decoder params from the pretrained model before we use it to initialize the current model. This means we have random decoder initialization.')
    parser.add_argument('--eliminate_embeddings_before_initialization', action='store_true', 
                        help='Lets wipe out the embedding params from the pretrained model before we use it to initialize the current model. This means we have random embedding initialization.')
    ### Distillation flags
    parser.add_argument('--distillation', action='store_true', 
                        help='Should we perform distillation from a parent model? If so then you must specify the model using "parent_pretrained_model". There are several distillation options check the flag called "distillation_styles".')
    parser.add_argument('--cross_distillation', action='store_true', 
                        help='Should we perform cross distillation from a parent model which has been trained on another source language but the same target language? If so then you must specify the model using "parent_pretrained_model". Additionally you should specify the train_src as a hyphen separated pair indicating the parent language and the child language. You should also ensure that the source file is a tab separated file where each line contains "the parent pair source sentence[tab]child pair source sentence" There are several distillation options check the flag called "distillation_styles".')
    parser.add_argument('--use_official_parent_pretrained', action='store_true', 
                        help='Use this flag if you want the argument "pretrained_model" to specify a pretrained model created by someone else for the purposes of distillation. Use this carefully because if the parent is created by someone else then you have to have your own model with different configurations for fine-tuning. Essentially you must make sure that use_official_parent_pretrained and use_official_pretrained are not true simultaneously.')
    parser.add_argument('--parent_pretrained_model', default='', type=str, 
                        help='Path to the parent pretrained model for distillation. The pretrained_model flag will be used to initialize the child model.')
    parser.add_argument('--distillation_loss_weight', type=float, default=0.7, 
                        help='All the distillation losses will be averaged and then multiplied by this weight before adding it to the regular xentropy loss which will be weighted by (1- distillation_loss_weight).')
    parser.add_argument('--distillation_styles', default='cross_entropy', type=str, 
                        help='One or more of softmax_distillation, attention_distillation, hidden_layer_regression. For attention distillation you must make sure that the number of attention heads between the parent and child are the same and for hidden layer regression you must make sure that the hidden size (d_model) is the same for the parent and child. In both these cases, you should also specify the layer mapping. See the "distillation_layer_mapping" flag.')
    parser.add_argument('--distillation_layer_mapping', default='1-1,2-2,3-3,4-4,5-5,6-6', type=str, 
                        help='This indicates the mappings between the parent and child model. The same flag is used for the encoder and the decoder. If you want to map the 2nd parent layer to the first child layer then use 2-1. Note that the layers are not zero indexed as per the description. Ensure that your indices are correct because checking is not done at the moment. If you get weird results then first make sure that your flags are correctly set. If the parent has 6 layers and the child has 3 layers then something like 6-4 will definitely throw an error. User beware! Dokuro mark.')
    parser.add_argument('--parent_encoder_layers', default=6, type=int, help="The value for number of encoder layers")
    parser.add_argument('--parent_decoder_layers', default=6, type=int, help="The value for number of decoder layers")
    parser.add_argument('--parent_dropout', default=0.1, type=float, help="The value for embedding dropout")
    parser.add_argument('--parent_attention_dropout', default=0.1, type=float, help="The value for attention dropout")
    parser.add_argument('--parent_activation_dropout', default=0.1, type=float, help="The value for activation dropout")
    parser.add_argument('--parent_encoder_attention_heads', default=8, type=int, help="The value for number of encoder attention heads")
    parser.add_argument('--parent_decoder_attention_heads', default=8, type=int, help="The value for number of decoder attention heads")
    parser.add_argument('--parent_decoder_ffn_dim', default=2048, type=int, help="The value for decoder ff hidden dim")
    parser.add_argument('--parent_encoder_ffn_dim', default=2048, type=int, help="The value for encoder ff hidden dim")
    parser.add_argument('--parent_d_model', default=512, type=int, help="The value for model hidden size")
    parser.add_argument('--save_weights_and_gradeint_info', action='store_true', 
                        help='Saving gradient information is time consuming. We should make this optional.')
    ###
    ### Placeholder flags to prevent code from breaking. These flags are not intended to be used for fine tuning. These flags are here because the common_utils.py methods assume the existence of these args for when joint mbart training and regular NMT training is done. TODO: Modify code to avoid the need for these flags in this script.
    parser.add_argument('--unify_encoder', action='store_true', 
                        help='Should we minimize the encoder representation distances instead of regular cross entropy minimization on the parallel corpus?')
    args = parser.parse_args()
    assert len(args.token_masking_probs_range) <= 2
    print("IP address is", args.ipaddr)
    
    args.world_size = args.gpus               #

    train_files = {}

    slangs = args.dev_slang.strip().split(",")
    tlangs = args.dev_tlang.strip().split(",")
    dev_srcs = args.dev_src.strip().split(",")
    dev_tgts = args.dev_tgt.strip().split(",")
    dev_files = {slang + "-" + tlang: (dev_src, dev_tgt) for slang, tlang, dev_src, dev_tgt in zip(slangs, tlangs, dev_srcs, dev_tgts)}
    print("Development files are:", dev_files)
    
    os.environ['MASTER_ADDR'] = args.ipaddr              #
    os.environ['MASTER_PORT'] = args.port
    model_create_load_run_save(args.gpus, args, train_files, dev_files)
    
if __name__ == "__main__":
    run_demo()