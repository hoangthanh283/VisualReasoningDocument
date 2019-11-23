import os
import argparse
import json
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

from preprocess.preprocess_document import (
    load_vocab, process_form_relation
)
from dataset import DocumentDataset
from model.relate_model import VQA

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args):
    vocab = load_vocab(args.vocab_path)
    dataset = DocumentDataset(args.input_dir, vocab, args.text_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = VQA(len(vocab["token_to_index"]), 256, 256, args.stack_size, args.text_len, DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.l2reg)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.gamma_size, gamma=0.1)
    criterion = nn.BCELoss()

    for epoch in range(args.num_epoch):
        print("\nEpoch %03d" % (epoch + 1))
        print("=========================")
        model.train()
        avg_loss = []
        
        for questions, vertex_vectors, edge_matrices, gt in loader:
            batch_size = questions.shape[0]

            questions = questions.long().to(DEVICE)
            questions_len = torch.full([batch_size], args.text_len).long().to(DEVICE)
            vertex_vectors = vertex_vectors.long().to(DEVICE)
            edge_matrices = edge_matrices.float().to(DEVICE)
            gt = gt.float().to(DEVICE)

            output, cache_module_prob, cache_attn = model(
                questions, questions_len, vertex_vectors, edge_matrices)
            loss = criterion(output, gt)
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())

        print("Epoch: {0} average loss: {1}".format(epoch, np.mean(avg_loss)))
        scheduler.step(epoch)

        if epoch == args.num_epoch - 1:
            save_checkpoint(epoch, model, optimizer, "weights/model.pth")
        print("Training finished!")


def save_checkpoint(epoch, model, optimizer, path):
    state = {
        "epoch": epoch,
        "model": model,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def test_output(args):
    # Init the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(args.vocab_path)
    print("============================")

    model = VQA(len(vocab["token_to_index"]), 256, 256, args.stack_size, args.text_len, device).to(device)
    model.eval()
    model.load_state_dict(torch.load("weights/model.pth")["state_dict"])

    # Process data
    path = "D:/Data/VQA/sanity/7_0.json"
    data = json.load(open(path, encoding="utf-8"))
    question, vertex_vectors, edge_matrices, mask, raw_nodes = process_position_json(data, vocab, get_raw=True)
    batch_size = 1

    question = torch.tensor(question).unsqueeze(0).long().to(device)
    question_len = torch.full([batch_size], args.text_len).long().to(device)
    vertex_vectors = torch.tensor(vertex_vectors).unsqueeze(0).long().to(device)
    edge_matrices = torch.tensor(edge_matrices).unsqueeze(0).float().to(device)

    # Run the model
    with torch.no_grad():
        model.infer(question, question_len, vertex_vectors, edge_matrices, raw_nodes)
    print("Finished")


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument("--input_dir", default="data_ibm_train_kv/training_data/annotations", type=str)
    parser.add_argument("--vocab_path", default="data_ibm_train_kv/charset.txt", type=str)
    parser.add_argument("--save_dir", default="weights")
    parser.add_argument("--text_len", type=int, default=200)
    parser.add_argument("--stack_size", type=int, default=4)
    # training parameters
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--l2reg", default=0.0, type=float)
    parser.add_argument("--num_epoch", default=500, type=int)
    parser.add_argument("--gamma_size", default=50, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--ratio", default=1, type=float, help="ratio of training examples")
    # model hyper-parameters
    args = parser.parse_args()
    is_train = True

    # make logging.info display into both shell and file
    os.makedirs(args.save_dir, exist_ok=True)

    # args display
    for k, v in vars(args).items():
        print("{0}: {1}".format(k, v))

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if is_train:
        train(args)
    test_output(args)


if __name__ == "__main__":
    main()
