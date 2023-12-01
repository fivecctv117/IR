import enum
import sys

sys.path += ['./']
import os
import torch
import gzip
import pickle
import subprocess
import csv
import multiprocessing
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
import json
from tqdm import tqdm
from star_tokenizer import RobertaTokenizer
import logging
import ast


def init_logging():
    handlers = [logging.StreamHandler()]
    handlers.append(logging.FileHandler("edit.log", mode="w"))
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    #logging.info("COMMAND: %s" % " ".join(sys.argv))

    
def pad_input_ids(input_ids, max_length,
                  pad_on_left=False,
                  pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids


def tokenize_to_file(args, in_path, output_dir, line_fn, max_length, begin_idx, end_idx):
    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case = True, cache_dir=None)
    os.makedirs(output_dir, exist_ok=True)
    data_cnt = end_idx - begin_idx
    ids_array = np.memmap(
        os.path.join(output_dir, "ids.memmap"),
        shape=(data_cnt, ), mode='w+', dtype=np.int32)
    token_ids_array = np.memmap(
        os.path.join(output_dir, "token_ids.memmap"),
        shape=(data_cnt, max_length), mode='w+', dtype=np.int32)
    token_length_array = np.memmap(
        os.path.join(output_dir, "lengths.memmap"),
        shape=(data_cnt, ), mode='w+', dtype=np.int32)
    pbar = tqdm(total=end_idx-begin_idx, desc=f"Tokenizing")
    for idx, line in enumerate(open(in_path, 'r')):
        if idx < begin_idx:
            continue
        if idx >= end_idx:
            break
        qid_or_pid, token_ids, length = line_fn(args, line, tokenizer)
        if length == 0:
            continue
        write_idx = idx - begin_idx
        ids_array[write_idx] = qid_or_pid
        token_ids_array[write_idx, :] = token_ids
        token_length_array[write_idx] = length
        pbar.update(1)
    pbar.close()
    logging.info("write_idx："+str(write_idx) )
    logging.info("data_cnt："+str(data_cnt) )

    #assert write_idx == data_cnt - 1


def multi_file_process(args, num_process, in_path, out_path, line_fn, max_length, token_fn):
    output_linecnt = subprocess.check_output(["wc", "-l", in_path]).decode("utf-8")
    print("line cnt", output_linecnt)
    all_linecnt = int(output_linecnt.split()[0])
    run_arguments = []
    for i in range(num_process):
        begin_idx = round(all_linecnt * i / num_process)
        end_idx = round(all_linecnt * (i+1) / num_process)
        output_dir = f"{out_path}_split_{i}"
        run_arguments.append((
                args, in_path, output_dir, line_fn,
                max_length, begin_idx, end_idx
            ))
    pool = multiprocessing.Pool(processes=num_process)
    pool.starmap(tokenize_to_file, run_arguments)
    pool.close()
    pool.join()
    splits_dir = [a[2] for a in run_arguments]
    return splits_dir, all_linecnt


def write_query_rel(args, pid2offset, qid2offset_file, query_file, positive_id_file, out_query_file, standard_qrel_file):

    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case = True, cache_dir=None)
    print( "Writing query files " + str(out_query_file) +
        " and " + str(standard_qrel_file))
    query_collection_path = os.path.join(args.data_dir,query_file)

    out_query_path = os.path.join(args.out_data_dir,out_query_file,)

    qid2offset = {}

   
    token_length_array = []

    with open(query_file, "r") as reader:
        input_data = json.load(reader)["questions"]
    valid_query_num = len(input_data) 
    token_ids_array = np.memmap(
        out_query_path+".memmap",
        shape=(valid_query_num, args.max_query_length), mode='w+', dtype=np.int32)
    qids = []
    qid_cur = 1000
    query_list = []
    qrel_docs = {}
    idx = 0
    
    print('start query file processing')
    for entry in input_data:
        qids.append(qid_cur)
        query_list.append(entry["body"])
        passage = tokenizer.encode(
            entry["body"].rstrip(),
            add_special_tokens=True,
            max_length=args.max_query_length,
            truncation=True)
        passage_len = min(len(passage), args.max_query_length)
        input_id_b = pad_input_ids(passage, args.max_query_length)
        #logging.info("idx, token" + str(idx) + str(input_id_b))
        token_ids_array[idx, :] = input_id_b
        #token_ids_array.append(input_id_b)
        token_length_array.append(passage_len) 
        #logging.info(passage_len)
        idx += 1
        for doc in entry["documents"]:
            qrel_docs[qid_cur] = doc[35:] #remove pmed url and left pid
    assert len(token_length_array) == len(token_ids_array) == idx
    np.save(out_query_path+"_length.npy", np.array(token_length_array))
        

    print("Total lines written: " + str(idx))
    meta = {'type': 'int32', 'total_number': idx,
            'embedding_size': args.max_query_length}
    with open(out_query_path + "_meta", 'w') as f:
        json.dump(meta, f)

    if positive_id_file is None:
        print("No qrels file provided")
        return
    print("Writing qrels")
    with open(os.path.join(args.out_data_dir, standard_qrel_file), "w", encoding='utf-8') as qrel_output: 
        out_line_count = 0
        rel = 1
        for qid in qrel_docs:
            qrel_output.write(str(qid) +
                         "\t0\t" + str(qrel_docs[qid]) +
                         "\t" + rel + "\n")
            out_line_count += 1
        print("Total lines written: " + str(out_line_count))

def build_bioqrel(sss):
    return 0


def preprocess(args):
    
    pid2offset = {}
    if args.data_type == 0:
        in_passage_path = os.path.join(
            args.data_dir,
            "allMeSH_limitjournals.json",
        )
    else:
        in_passage_path = os.path.join(
            args.data_dir,
            "collection.tsv",
        )

    out_passage_path = os.path.join(
        args.out_data_dir,
        "bio-passages",
    )

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return

    print('start passage file split processing')
    splits_dir_lst, all_linecnt = multi_file_process(
        args, args.threads, in_passage_path,
        out_passage_path, PassagePreprocessingFn,
        args.max_seq_length, tokenize_to_file
        )

    token_ids_array = np.memmap(
        out_passage_path+".memmap",
        shape=(all_linecnt, args.max_seq_length), mode='w+', dtype=np.int32)
    token_length_array = []

    idx = 0
    out_line_count = 0
    print('start merging splits')
    for split_dir in splits_dir_lst:
        ids_array = np.memmap(
            os.path.join(split_dir, "ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = np.memmap(
            os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = split_token_ids_array.reshape(len(ids_array), -1)
        split_token_length_array = np.memmap(
            os.path.join(split_dir, "lengths.memmap"), mode='r', dtype=np.int32)
        for p_id, token_ids, length in zip(ids_array, split_token_ids_array, split_token_length_array):
            token_ids_array[idx, :] = token_ids
            token_length_array.append(length) 
            pid2offset[p_id] = idx
            idx += 1
            if idx < 3:
                print(str(idx) + " " + str(p_id))
            out_line_count += 1
    assert len(token_length_array) == len(token_ids_array) == idx
    np.save(out_passage_path+"_length.npy", np.array(token_length_array))

    print("Total lines written: " + str(out_line_count))
    meta = {
        'type': 'int32',
        'total_number': out_line_count,
        'embedding_size': args.max_seq_length}
    with open(out_passage_path + "_meta", 'w') as f:
        json.dump(meta, f)
    # objects = []
    # with open("data/doc/preprocess/pid2offset.pickle", "rb") as f:
    #     objects = pickle.load(f)
    # logging.info(objects)
    pid2offset_path = os.path.join(
        args.out_data_dir,
        "pid2offset.pickle",
    )
    with open(pid2offset_path, 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4)
    
    print("done saving pid2offset")
    
    if args.data_type == 0:
         in_query_path = os.path.join(
            args.data_dir,
            "BioASQ-trainingDataset5b.json")
        
         write_query_rel(
            args,
            pid2offset,
            "train-qid2offset.pickle",
            in_query_path,
             None,
            "bio-train-query",
             "bio-train-qrel.tsv")
            
def PassagePreprocessingFn(args, line, tokenizer):
    if args.data_type == 0:
        #logging.info("new line fuc")
        #logging.info(line)
        line = line.rstrip()
        line_arr = {}
        #logging.info(line)

        if line.endswith(","):
            line = line[:-1]
        if "abstractText" in line: 
            if line.count('\'') >= 6:
                #line = line.replace("\'", "\"")
                line_arr = ast.literal_eval(line)
            else:
                line_arr = json.loads(line)
        else:
            return 0,0,0
 
        if line_arr == "":
            logging.info("Error line: " + line)
        p_id = int(line_arr["pmid"])  # remove "D"

        #url = line_arr[1].rstrip()
        title = line_arr["title"].rstrip()
        p_text = line_arr["abstractText"].rstrip()
        # NOTE: This linke is copied from ANCE, 
        # but I think it's better to use <s> as the separator, 
        full_text = title + "<sep>" + p_text
        #logging.info(full_text)

        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = full_text[:args.max_doc_character]
    else:
        line = line.strip()
        line_arr = line.split('\t')
        p_id = int(line_arr[0])

        p_text = line_arr[1].rstrip()
        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = p_text[:args.max_doc_character]
    #logging.info(full_text)
    passage = tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=args.max_seq_length,
        truncation=True
    )
    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length)

    return p_id, input_id_b, passage_len


def BioQueryPreprocessingFn(args, input_file, tokenizer):
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["questions"]
    qids = []
    qid_cur = 1000
    query_list = []
    qrel_docs = {}
    for entry in input_data:
        qids.append(qid_cur)
        query_list.append(entry["body"])
        passage = tokenizer.encode(
            line_arr[1].rstrip(),
            add_special_tokens=True,
            max_length=args.max_query_length,
            truncation=True)
        passage_len = min(len(passage), args.max_query_length)
        input_id_b = pad_input_ids(passage, args.max_query_length)

        
        for doc in entry["documents"]:
            qrel_docs[qid_cur] = doc[35:] #remove pmed url and left pid
            
            
            
    logging.info(line)
    line_arr = line.split('\t')
    q_id = int(line_arr[0])

    passage = tokenizer.encode(
        line_arr[1].rstrip(),
        add_special_tokens=True,
        max_length=args.max_query_length,
        truncation=True)
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)

    return q_id, input_id_b, passage_len


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="roberta-base",
        type=str,
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_doc_character",
        default=10000,
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )
    parser.add_argument(
        "--data_type",
        default=1,
        type=int,
        help="0 for doc, 1 for passage",
    )
    parser.add_argument("--threads", type=int, default=32)

    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    if args.data_type == 0:
        args.data_dir = "./data/doc/dataset"
        args.out_data_dir = "./data/doc/preprocess"
    else:
        args.data_dir = "./data/passage/dataset"
        args.out_data_dir = "./data/passage/preprocess"

    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    init_logging();
    logging.info("work");
    preprocess(args)


if __name__ == '__main__':
    main()
