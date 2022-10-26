# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import ast
import csv
import json
import logging
import os
import sys
from io import open
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
import torch

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ast_dict = []

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, code_tokens, code_ids, code_ast, nl_tokens, nl_ids, label, idx):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.code_ast = code_ast
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.label = label
        self.idx = idx


class InputFeaturesTriplet(InputFeatures):
    """A single training/test features for a example. Add docstring seperately. """
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, ds_tokens, ds_ids, label, idx):
        super(InputFeaturesTriplet, self).__init__(code_tokens, code_ids, nl_tokens, nl_ids, label, idx)
        self.ds_tokens = ds_tokens
        self.ds_ids = ds_ids


def convert_examples_to_features(js, tokenizer, args, ast):
    # label
    label = js['label']

    # code
    code = js['code']
    code_tokens = tokenizer.tokenize(code)[:args.max_seq_length-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.max_seq_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    # ast
    code_ast = ast
    padding_length = args.max_seq_length - len(code_ast)
    # TODO: Check which padding token makes sense
    code_ast += [1]*padding_length

    # query
    nl = js['doc']
    nl_tokens = tokenizer.tokenize(nl)[:args.max_seq_length-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.max_seq_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(code_tokens, code_ids, code_ast, nl_tokens, nl_ids, label, js['idx'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, type=None):
        # json file: dict: idx, query, doc, code
        self.examples = []
        self.type = type

        logger.info(f"Loading data from {file_path}")

        data=[]
        with open(file_path, 'r') as f:
            data = json.load(f)
        if self.type == 'test':
            for js in data:
                js['label'] = 0
        with open('./ast_dict.json', 'r') as f:
            if f.read(1):
                f.seek(0)
                ast_list = json.load(f)
                for e in ast_list:
                    if ast_dict.count(e) == 0:
                        ast_dict.append(e)
            f.close()
        for js in data:
            try:
                code_ast = []
                for node in ast.walk(ast.parse(js['code'])):
                    name = node.__class__.__name__
                    if ast_dict.count(name) == 0:
                        ast_dict.append(name)
                    code_ast.append(ast_dict.index(name) + 2)
                if(len(code_ast) <= 200):
                    self.examples.append(convert_examples_to_features(js, tokenizer, args, code_ast))
                
            except:
                continue

        with open('ast_dict.json', 'w') as f:
            json.dump(ast_dict, f)
            f.close()
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return (torch.tensor(self.examples[i].code_ids),
                torch.tensor(self.examples[i].code_ast),
                torch.tensor(self.examples[i].nl_ids),
                torch.tensor(self.examples[i].label))




def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    prec = precision_score(y_true=labels, y_pred=preds)
    reca = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": prec,
        "recall": reca,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "webquery":
        return acc_and_f1(preds, labels)
    if task_name == "staqc":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

