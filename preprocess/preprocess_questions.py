#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json

import numpy as np
import pickle

from exp_document.preprocess import programs
from exp_document.preprocess.utils import tokenize, encode, build_vocab


"""
Preprocessing script for CLEVR question files.
"""


def program_to_str(program, mode):
    if mode == "chain":
        if not programs.is_chain(program):
            return None
    elif mode == "prefix":
        program = programs.list_to_prefix(program)
    elif mode == "postfix":
        program = programs.list_to_postfix(program)

    # Convert to program and inputs
    for f in program:
        if f["function"] in {"equal_shape", "equal_color", "equal_size", "equal_material"}:
            f["function"] = "equal"
        elif "query" in f["function"]:
            # <cat> of query_<cat>
            value = f["function"][6:]
            f["function"] = "query"
            f["value_inputs"].append(value)
            assert len(f["value_inputs"]) == 1
        elif "same" in f["function"]:
            # <cat> of same_<cat>
            value = f["function"][5:]
            f["function"] = "same"
            f["value_inputs"].append(value)
        elif "filter_" in f["function"]:
            f["function"] = "filter"

        if len(f["value_inputs"]) == 0:
            f["value_inputs"].append("<PAD>")
        assert len(f["value_inputs"]) == 1

    func_str = " ".join(f["function"] for f in program)
    input_str = " ".join(f["value_inputs"][0] for f in program)
    return func_str, input_str


def main(args):
    questions = []
    vocab = {}

    # Encode all questions and programs
    print("Encoding data")
    questions_encoded = []
    programs_encoded = []
    # value_inputs, encoded by token_to_index in CLEVR
    # because all valid inputs are in question vocab
    program_inputs_encoded = []
    question_families = []
    orig_indices = []
    image_indices = []
    answers = []

    for orig_index, q in enumerate(questions):
        question = q["question"]

        orig_indices.append(orig_index)
        image_indices.append(q["image_index"])
        question_tokens = tokenize(
            question, punct_to_keep=[";", ","], punct_to_remove=["?", "."])
        question_encoded = encode(
            question_tokens, vocab["token_to_index"], allow_unk=args.encode_unk == 1)
        questions_encoded.append(question_encoded)

        if "program" in q:
            program = q["program"]
            program_str, input_str = program_to_str(program, args.mode)
            program_tokens = tokenize(program_str)
            program_encoded = encode(program_tokens, vocab["program_token_to_idx"])
            programs_encoded.append(program_encoded)
            # program value_inputs
            input_tokens = tokenize(input_str)
            input_encoded = encode(input_tokens, vocab["token_to_index"])
            assert len(input_encoded) == len(program_encoded)  # input should have the same len with func
            program_inputs_encoded.append(input_encoded)

        if "answer" in q:
            answers.append(vocab["answer_token_to_idx"][q["answer"]])

    # Pad encoded questions and programs
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab["token_to_index"]["<PAD>"])

    if len(programs_encoded) > 0:
        max_program_length = max(len(x) for x in programs_encoded)
        for pe in programs_encoded:
            while len(pe) < max_program_length:
                pe.append(vocab["program_token_to_idx"]["<PAD>"])
        for ie in program_inputs_encoded:
            while len(ie) < max_program_length:
                ie.append(vocab["token_to_index"]["<PAD>"])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    programs_encoded = np.asarray(programs_encoded, dtype=np.int32)
    program_inputs_encoded = np.asarray(program_inputs_encoded, dtype=np.int32)
    print(questions_encoded.shape)
    print(programs_encoded.shape)
    print(program_inputs_encoded.shape)
    print("Writing")
    data = {
        "questions": questions_encoded,
        "image_indices": np.asarray(image_indices),
        "orig_indices": np.asarray(orig_indices),
        "programs": programs_encoded,
        "program_inputs": program_inputs_encoded,
        "question_families": question_families,
        "answers": answers,
    }
    with open(args.output_pt_file, "wb") as f:
        pickle.dump(data, f)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="prefix",
                        choices=["chain", "prefix", "postfix"])
    parser.add_argument("--input_vocab_json", default="")
    parser.add_argument("--expand_vocab", default=0, type=int)
    parser.add_argument("--unk_threshold", default=1, type=int)
    parser.add_argument("--encode_unk", default=0, type=int)

    parser.add_argument("--input_questions_json", required=True)
    parser.add_argument("--output_pt_file", required=True)
    parser.add_argument("--output_vocab_json", default="")
    arguments = parser.parse_args()
    main(arguments)
