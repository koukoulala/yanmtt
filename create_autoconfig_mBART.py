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

from transformers import AutoConfig, AlbertTokenizer, AutoTokenizer, MBartTokenizer
import sys
import os
import sentencepiece as spm

src_dir = sys.argv[1] # Has to be a comma separated list
vocab_size = sys.argv[2]
tgt_folder = sys.argv[3]
type = sys.argv[4]
user_tokens = sys.argv[5]
num_train_sentences = sys.argv[6]
character_coverage = sys.argv[7]

os.makedirs(tgt_folder, exist_ok=True)
src_files = ",".join([os.path.join(src_dir, file_name) for file_name in os.listdir(src_dir)])
print("src_files", src_files)

if type == "mbart":
    spm.SentencePieceTrainer.train(max_sentence_length=20000, input=src_files, model_prefix=os.path.join(tgt_folder, "sentencepiece.bpe"),
                               vocab_size=vocab_size, pad_id=0, unk_id=1, eos_id=-1, bos_id=-1, user_defined_symbols="[CLS],[SEP],[MASK]",
                               shuffle_input_sentence=True, character_coverage=character_coverage, model_type="bpe", input_sentence_size=num_train_sentences)
else:
    print("Unknown tokenizer. Exiting!")
    sys.exit(1)

if type == "albert":
    tokenizer = AlbertTokenizer.from_pretrained(tgt_folder, do_lower_case=False, use_fast=False, keep_accents=True, strip_accents=False)
elif type == "mbart":
    tokenizer = MBartTokenizer.from_pretrained(tgt_folder, do_lower_case=False, use_fast=False, keep_accents=True, strip_accents=False)
else:
    print("Unknown tokenizer. Exiting!")
    sys.exit(1)

special_tokens_dict = {'additional_special_tokens': ["<s>", "</s>"] + (user_tokens.strip().split(",") if user_tokens is not "." else [])} ## Add additional special tokens specified by the user as a comma separated list.

for lang_file in src_files.strip().split(","):
    lang_tok = lang_file.strip().split(".")[-1] ## Asuuming that the file extension indicates the tgt language
    if "<2"+lang_tok+">" not in special_tokens_dict["additional_special_tokens"]:
        special_tokens_dict["additional_special_tokens"].append("<2"+lang_tok+">")

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.save_pretrained(tgt_folder)

os.rename(os.path.join(tgt_folder, "tokenizer_config.json"), os.path.join(tgt_folder, "config.json"))

config = AutoConfig.from_pretrained(tgt_folder)
config.save_pretrained(tgt_folder)

print("Testing tokenizer")

tokenizer = AutoTokenizer.from_pretrained(tgt_folder, do_lower_case=False, use_fast=False, keep_accents=True, strip_accents=False)

print(tokenizer)

with open("data/sample.all", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        print(i, tokenizer.convert_ids_to_tokens(tokenizer(line, add_special_tokens=False).input_ids))

# print(tokenizer.convert_ids_to_tokens(tokenizer("This is a dummy sentence. Depending on the languages you chose for segmentation, this may or may not look weirdly segmented to you.", add_special_tokens=False).input_ids))
    
print("Success")
