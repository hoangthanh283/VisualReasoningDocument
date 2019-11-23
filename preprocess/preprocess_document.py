from __future__ import print_function, division

import os
import glob
import cv2
import random
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt 


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_LEN_TEXT = 200
NUM_EDGES = 5 # numbers of edegs: left, top, right, bottom, linking
DISTANCE = 10 # Distance to find neighbors
NODE_LABELS = {"header": 0, "question": 1, "answer": 2, "other": 3}
RELATIONS = ["left", "top", "right", "bottom", "linking"]
QUESTION_TEMPLATES = ["is this", "which lines are"]


def load_vocab(vocab_path):
    vocab = dict()
    vocab["token_to_index"] = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    with open(vocab_path, encoding="utf-8") as file:
        # line = file.read().strip()
        line = file.read()
        characters = list(line)

    vocab_count = max([v for k, v in vocab["token_to_index"].items()])
    for c in characters:
        vocab["token_to_index"][c] = vocab_count + 1
        vocab_count += 1

    vocab["index_to_token"] = {v: k for k, v in vocab["token_to_index"].items()}
    return vocab

def get_node_attributes(region, vocab, text_len=10):
    """ return node encoding and raw value text of the node
    Args:
        region: node dict, {box: [], label: '', linking: []}
        vocab: dict of character and their ids
        text_len: max len of text line to encode
    Return:
        features (list): [4 node positions, text encodes, label, type_kv, id]
        tex (str): text label of the node
    """
    _id = region["id"]

    # get list of value candidates
    list_value_id = list(map(lambda l: l[1], region["linking"]))

    # get list of key candidates
    list_key_id = list(map(lambda l: l[0], region["linking"]))

    # define key-value entity
    if _id in list_key_id:
        _type = 1 # key
    elif _id in list_value_id:
        _type = 2 # value
    else:
        _type = 0 # other

    # define onehot encode node label 
    onehot_label = np.eye(len(NODE_LABELS))[NODE_LABELS[region["label"]]]

    # encode the node features includes: node position, 
    # text embedding, node label, type kv, id node
    features = [0] * 4 + [vocab["token_to_index"][PAD_TOKEN]] * text_len + \
        list(map(int, onehot_label)) + [_type, _id]

    left, top, right, bottom = region.get("box", [0, 0, 0, 0])
    features[:4] = [left, top, right, bottom]

    #text = region["region_attributes"]["label"]
    text = region.get("text", "")
    for index in range(text_len):
        if index < len(text):
            features[index + 4] = vocab["token_to_index"].get(
                text[index], vocab["token_to_index"][UNK_TOKEN])
    return features, text

def get_relation_adjacency(vertex_vectors, regions):
    """ create direct adjacency matrix """

    vertex_vector_map = {vector[-1]: vector for vector in vertex_vectors}
    adjacency_matrix = np.zeros([len(vertex_vectors), len(vertex_vectors), NUM_EDGES])

    for region in regions:
        _id = region["id"]
        linkings = region["linking"]
        for link in linkings:
            if (adjacency_matrix[link[0], link[1], -1] == 0 \
                or adjacency_matrix[link[1], link[0], -1] == 0):
                adjacency_matrix[link[0], link[1], -1] = 1

        neighbors = find_neighbors(vertex_vector_map[_id], vertex_vectors)
        for j, n in enumerate(neighbors):
            if n is not None:
                adjacency_matrix[_id, n, :4] = np.eye(4)[j] # as 4x1 adjacency matrix as 4 neighbors

    return adjacency_matrix

def find_neighbors(vertex, vertex_vectors):
    """ return the index of four nearest neighbors of the given entity
        if no neigbor, return None, Order: left, top, right, bottom
        Eg: (left, top, right, bottom) = (0, 1, None, 3)
    """

    def same_row(item):
        # Check all items are on the same row 
        if abs(vertex[1] - item[1]) < DISTANCE:
            return True
        if abs(vertex[3] - item[3]) < DISTANCE:
            return True
        return False

    def same_column(item):
        # Check all items are on the same column
        if abs(vertex[0] - item[0]) < DISTANCE:
            return True
        if abs(vertex[2] - item[2]) < DISTANCE:
            return True
        return False

    # Check the given entity position is valid or not
    assert vertex[2] > vertex[0] and vertex[3] > vertex[1]

    # Get all entities on the same row (horizontal)
    candidates = [v for v in vertex_vectors if same_row(v)]

    # Define left entities of all horizontal entities, get the nearest one
    left_candidates = np.array([c for c in candidates if c[2] < vertex[0]])
    left_neighbor = None
    if len(left_candidates) > 0:
        left_index = np.argmax(left_candidates[:, 2])
        left_neighbor = left_candidates[left_index][-1]

    # Define right entities of all horizontal entities, get the nearest one
    right_candidates = np.array([c for c in candidates if c[0] > vertex[2]])
    right_neighbor = None
    if len(right_candidates) > 0:
        right_index = np.argmin(right_candidates[:, 0])
        right_neighbor = right_candidates[right_index][-1]

    # Get all entities on the same column (vertical)
    candidates = [v for v in vertex_vectors if same_column(v)]

    # Define top entities of all vertical entities, get the nearest one
    top_candidates = np.array([c for c in candidates if c[3] < vertex[1]])
    top_neighbor = None
    if len(top_candidates) > 0:
        top_index = np.argmax(top_candidates[:, 3])
        top_neighbor = top_candidates[top_index][-1]

    # Define bottom entities of all vertical entitie, get the nearest one 
    bottom_candidates = np.array([c for c in candidates if c[1] > vertex[3]])
    bottom_neighbor = None
    if len(bottom_candidates) > 0:
        bottom_index = np.argmin(bottom_candidates[:, 1])
        bottom_neighbor = bottom_candidates[bottom_index][-1]

    return (left_neighbor, top_neighbor, right_neighbor, bottom_neighbor)

def show_graph(image, vertex_vectors, relation_matrix):
    if image is None:
        return
    
    # define adj dict 
    vertex_vector_map = {vector[-1]: vector for vector in vertex_vectors}

    # visual adjacency nodes 
    for node_id, node_vector in vertex_vector_map.items():
        vertex_center = ((node_vector[0] + node_vector[2]) // 2, \
            (node_vector[1] + node_vector[3]) // 2)
        cv2.rectangle(image, (node_vector[0], node_vector[1]), \
            (node_vector[2], node_vector[3]), (255, 0, 0), 2)
        
        for adj_id in range(0, relation_matrix.shape[0]):
            relative_positions = relation_matrix[node_id][adj_id][:NUM_EDGES-1] 
            if relation_matrix[node_id][adj_id][-1] == 1: # linking
                adj_node_center = ((vertex_vector_map[adj_id][0] + vertex_vector_map[adj_id][2]) // 2, \
                    (vertex_vector_map[adj_id][1] + vertex_vector_map[adj_id][3]) // 2)
                cv2.line(image, vertex_center, adj_node_center, (0, 255, 0), 2)

            neighbor_vector = vertex_vector_map[adj_id]
            if relative_positions[1] == 1 or relative_positions[3] == 1: # top or bottom
                y = (neighbor_vector[1] + neighbor_vector[3]) // 2
                cv2.line(image, vertex_center, (vertex_center[0], y), (0, 0, 255), 1)
            elif relative_positions[0] == 1 or relative_positions[2] == 1: # left or right
                x = (neighbor_vector[0] + neighbor_vector[2]) // 2
                cv2.line(image, vertex_center, (x, vertex_center[1]), (0, 0, 255), 1)

    plt.imshow(image)
    plt.show()

def refine_node_features(vertex_vectors):
    # remove position of node
    # vertex_vectors = vertex_vectors[:, 4:-1]
    # vertex_vectors = vertex_vectors[:, :-(len(NODE_LABELS)+2)]
    vertex_vectors = vertex_vectors[:, 4:-(len(NODE_LABELS)+2)]
    return vertex_vectors

def encode_bow(texts, vocab):
    vocab_dict = vocab["token_to_index"]
    bag = np.zeros([len(texts), len(vocab_dict)])
    for index, text in enumerate(texts):
        for char in text:
            char_index = int(vocab_dict[char])
            bag[index, char_index] += 1
    return bag

def process_form_relation(data, vocab, max_text_length, debug=False):
    vertex_vectors = []
    raw_nodes = []
    regions = data["form"]

    # get all verteces
    vertex_vectors = []
    regions = data["form"]
    for index, region in enumerate(regions):
        vertex, raw_node = get_node_attributes(region, vocab, text_len=max_text_length)
        vertex_vectors.append(vertex)
        raw_nodes.append(raw_node)

    raw_nodes = np.array(raw_nodes)
    vertex_vectors = np.array(vertex_vectors)
    vertex_vector_map = {vector[-1]: vector for vector in vertex_vectors}

    # get relation adjacency matrix
    relation_matrix = get_relation_adjacency(vertex_vectors, regions)

    # get a randomly node relation
    node_index = random.randrange(0, len(vertex_vectors))
    neighbors = relation_matrix[node_index]

    # generate a randomly encoded question 
    keywords = list(NODE_LABELS.keys())
    random_key = random.randint(0, len(keywords)-1)
    random_relation = random.randint(0, len(RELATIONS)-1)

    k_question = "{0} {1} a {2}?".format(QUESTION_TEMPLATES[0], \
        raw_nodes[node_index], keywords[random_key])
    r_question = "{0} {1} of {2}".format(QUESTION_TEMPLATES[1], \
        RELATIONS[random_relation], raw_nodes[node_index])
    
    choose_question = random.choice([("class", k_question, random_key), \
        ("relation", r_question, random_relation)])
    _type, raw_question, random_position = choose_question

    question = [vocab["token_to_index"].get(
        c, vocab["token_to_index"][UNK_TOKEN]) for c in raw_question]
    question = question + [vocab["token_to_index"][PAD_TOKEN]] * max(max_text_length - len(question), 0)
    question = np.array(question)

    """
    keywords = ["左", "上", "右", "下"]
    random_position = random.randint(0, 3)
    random_position = np.eye(4)[random_position]
    raw_question = "%sの'%s'は?" % (keywords[random_position], raw_nodes[node_index])
    vertex_vectors = encode_bow(raw_nodes, vocab)
    question = vertex_vectors[node_index]
    """

    # node mask label, has 2 channels, relations & node classes
    mask = np.zeros([len(vertex_vectors) + len(NODE_LABELS) + 1])
    actual_node_class = vertex_vector_map[node_index][-(2+len(NODE_LABELS)): -2]
    mask[1: len(NODE_LABELS)+1] = actual_node_class
    if _type == "relation":
        for index in range(0, len(vertex_vectors)):
            if neighbors[index, random_position] == 1:
                mask[index + len(NODE_LABELS) + 1] = 1
    elif _type == "class":
        if np.argmax(actual_node_class) == random_position:
            mask[0] = 1
        else:
            mask[0] = 0

    if debug == True:
        return (raw_question, vertex_vectors, relation_matrix, mask, raw_nodes)
    return (question, refine_node_features(vertex_vectors), relation_matrix, mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./data_ibm_train_kv/training_data/images", \
        help="folder path that contain image files")
    parser.add_argument("--label_dir", default="./data_ibm_train_kv/training_data/annotations", \
        help="folder path that contain json label files")
    parser.add_argument("--vocab", default="./data_ibm_train_kv/charset.txt", help="vocab file path")
    args = parser.parse_args()

    # pick randomly and load a json label file, image file
    json_filename = random.choice(os.listdir(args.label_dir))
    json_file_path = os.path.join(args.label_dir, json_filename)
    image_filename = [f for f in os.listdir(args.image_dir) \
        if os.path.splitext(json_filename)[0] in f]

    # load vocab and read image wrt label file
    with open(json_file_path, encoding="utf-8") as file:
        data = json.load(file)
    vocab = load_vocab(args.vocab)

    if image_filename is not None:
        image_file_path = os.path.join(args.image_dir, image_filename[-1])
        image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

        raw_question, vertex_vectors, relation_matrix, mask, raw_nodes = \
            process_form_relation(data, vocab, MAX_LEN_TEXT, debug=True)

        # visualize realtion and edge of all nodes
        # print(raw_nodes)
        print("Question: {0}".format(raw_question))
        show_graph(image, vertex_vectors, relation_matrix)


if __name__ == "__main__":
    main()
