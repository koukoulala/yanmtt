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

additional_tokens = sys.argv[4].strip().split(",") if sys.argv[4] is not "." else []
for lang_file in sys.argv[3].strip().split(","):
    lang_tok=lang_file.strip().split(".")[-1] ## Asuuming that the file extension indicates the tgt language
    if "<2"+lang_tok+">" not in additional_tokens:
        additional_tokens.append("<2"+lang_tok+">")

if sys.argv[2] == "albert":
    tokenizer = AlbertTokenizer.from_pretrained(sys.argv[1], do_lower_case=False, use_fast=False, keep_accents=True, strip_accents=False)
    special_tokens_dict = {'additional_special_tokens': ["<s>", "</s>"] + additional_tokens} ## Add additional special tokens specified by the user as a comma separated list.
    tokenizer.add_special_tokens(special_tokens_dict) ## This craps out for mbart tokenizer as in it does not allocate separate token ids for the special tokens. We will need to do a lot of manual work to fix this.
elif sys.argv[2] == "mbart":
    additional_tokens.extend(["[MASK]", "[CLS]", "[SEP]"]) ## Add the special tokens to the additional tokens list. They may not be used irl but keeping them here just in case. Note that <s> and </s> are already added by default in the implementation.
    with open(os.path.join(sys.argv[1], "specially_added_tokens"), "w") as f:
        f.write("\n".join(additional_tokens))
    tokenizer = MBartTokenizer.from_pretrained(sys.argv[1], do_lower_case=False, use_fast=False, keep_accents=True, strip_accents=False)
else:
    print("Unknown tokenizer. Exiting!")
    sys.exit(1)

tokenizer.save_pretrained(sys.argv[1])

os.rename(sys.argv[1]+"/tokenizer_config.json", sys.argv[1]+"/config.json")

config = AutoConfig.from_pretrained(sys.argv[1])
config.save_pretrained(sys.argv[1])

print("Testing tokenizer")

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], do_lower_case=False, use_fast=False, keep_accents=True, strip_accents=False)

print(tokenizer)

with open("data/sample.all", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        print(i, tokenizer.convert_ids_to_tokens(tokenizer(line, add_special_tokens=False).input_ids))

# print(tokenizer.convert_ids_to_tokens(tokenizer("This is a dummy sentence. Depending on the languages you chose for segmentation, this may or may not look weirdly segmented to you.", add_special_tokens=False).input_ids))
    
print("Success")
