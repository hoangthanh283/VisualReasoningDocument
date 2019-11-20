import random
import json
import argparse
import numpy as np
import cv2


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_LEN_TEXT = 300

def load_vocab(vocab_path):
    vocab = dict()
    vocab["token_to_index"] = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    with open(vocab_path, encoding="utf-8") as file:
        line = file.read().strip()
        characters = list(line)

    vocab_count = max([v for k, v in vocab["token_to_index"].items()])
    for c in characters:
        vocab["token_to_index"][c] = vocab_count + 1
        vocab_count += 1

    vocab["index_to_token"] = {v: k for k, v in vocab["token_to_index"].items()}
    return vocab


def get_true_node_attributes(region, vocab, index):
    features = [0, 0, 0, 0] + [vocab["token_to_index"][PAD_TOKEN]] * 12 + [index]

    shape = region["shape_attributes"]
    if "x" in shape:
        left, width = shape["x"], shape["width"]
        top, height = shape["y"], shape["height"]
        right = left + width
        bottom = top + height
    else:
        left, right = min(shape["all_points_x"]), max(shape["all_points_x"])
        top, bottom = min(shape["all_points_y"]), max(shape["all_points_y"])
    features[:4] = [left, top, right, bottom]

    text = region["region_attributes"]["label"]
    for index in range(12):
        if index < len(text):
            features[index + 4] = vocab["token_to_index"].get(
                text[index], vocab["token_to_index"][UNK_TOKEN])
    return features


def get_node_attributes(region, vocab, index, text_len=10):
    features = [0] * 4 + [vocab["token_to_index"][PAD_TOKEN]] * text_len + [index]
    #shape = region["shape_attributes"]
    shape = region.get("box", [0, 0, 0, 0])
    left, top, width, height = shape
    right, bottom = left + width, top + height
    """
    if "x" in shape:
        left, width = shape["x"], shape["width"]
        top, height = shape["y"], shape["height"]
        right = left + width
        bottom = top + height
    else:
        left, right = min(shape["all_points_x"]), max(shape["all_points_x"])
        top, bottom = min(shape["all_points_y"]), max(shape["all_points_y"])
    """
    features[:4] = [left, top, right, bottom]

    #text = region["region_attributes"]["label"]
    text = region.get("text", "")
    for index in range(text_len):
        if index < len(text):
            features[index + 4] = vocab["token_to_index"].get(
                text[index], vocab["token_to_index"][UNK_TOKEN])
    return features, text


def find_neighbors(vertex, vertex_vectors):
    def same_row(item):
        if abs(vertex[1] - item[1]) < 10:
            return True
        if abs(vertex[3] - item[3]) < 10:
            return True
        return False

    def same_column(item):
        if abs(vertex[0] - item[0]) < 10:
            return True
        if abs(vertex[2] - item[2]) < 10:
            return True
        return False

    assert vertex[2] > vertex[0] and vertex[3] > vertex[1]
    candidates = [v for v in vertex_vectors if same_row(v)]

    left_candidates = np.array([c for c in candidates if c[2] < vertex[0]])
    left_neighbor = None
    if len(left_candidates) > 0:
        left_index = np.argmax(left_candidates[:, 2])
        left_neighbor = left_candidates[left_index][-1]

    right_candidates = np.array([c for c in candidates if c[0] > vertex[2]])
    right_neighbor = None
    if len(right_candidates) > 0:
        right_index = np.argmin(right_candidates[:, 0])
        right_neighbor = right_candidates[right_index][-1]

    candidates = [v for v in vertex_vectors if same_column(v)]

    top_candidates = np.array([c for c in candidates if c[3] < vertex[1]])
    top_neighbor = None
    if len(top_candidates) > 0:
        top_index = np.argmax(top_candidates[:, 3])
        top_neighbor = top_candidates[top_index][-1]

    bottom_candidates = np.array([c for c in candidates if c[1] > vertex[3]])
    bottom_neighbor = None
    if len(bottom_candidates) > 0:
        bottom_index = np.argmin(bottom_candidates[:, 1])
        bottom_neighbor = bottom_candidates[bottom_index][-1]

    return left_neighbor, top_neighbor, right_neighbor, bottom_neighbor


def show_graph(image, vertex_vectors, edge_matrices):
    if image is None:
        return

    for index, vertex in enumerate(vertex_vectors):
        cv2.rectangle(image, (vertex[0], vertex[1]), (vertex[2], vertex[3]), (255, 0, 0), 2)

        for (i, j), label in edge_matrices.items():
            if i != index:
                continue

            center_a = ((vertex[0] + vertex[2]) // 2, (vertex[1] + vertex[3]) // 2)
            neighbor_index = i if j == index else j
            neighbor = vertex_vectors[neighbor_index]
            if label[1] == 1 or label[3] == 1:
                y = (neighbor[1] + neighbor[3]) // 2
                cv2.line(image, center_a, (center_a[0], y), (0, 0, 255), 2)
            else:
                x = (neighbor[0] + neighbor[2]) // 2
                cv2.line(image, center_a, (x, center_a[1]), (0, 0, 255), 2)
    cv2.imwrite("output.png", image)


def refine_node_features(vertex_vectors):
    vertex_vectors = vertex_vectors[:, 4:-1]
    return vertex_vectors


def encode_bow(texts, vocab):
    bag = np.zeros([len(texts), len(vocab)])
    for index, text in enumerate(texts):
        for char in text:
            char_index = vocab.index(char)
            bag[index, char_index] += 1
    return bag


def process_document_json(data, vocab, get_raw=False):
    vertex_vectors = []
    raw_nodes = []
    regions = data["attributes"]["_via_img_metadata"]["regions"]
    for index, region in enumerate(regions):
        vertex, raw_node = get_node_attributes(region, vocab, index, text_len=16)
        vertex_vectors.append(vertex)
        raw_nodes.append(raw_node)
    vertex_vectors = np.array(vertex_vectors)
    raw_nodes = np.array(raw_nodes)

    edge_matrices = np.zeros([len(vertex_vectors), len(vertex_vectors)])
    for index, vertex in enumerate(vertex_vectors):
        neighbors = find_neighbors(vertex, vertex_vectors)
        for j, n in enumerate(neighbors):
            if n is not None:
                edge_matrices[index, n] = 1

    keywords = {
        "tel": "TEL",
        "fax": "FAX",
        "company_name": "会社",
        "amount_including_tax": "合計",
    }
    target = random.choice(["tel", "fax"])
    raw_question = "%sは?" % keywords[target]
    question = [vocab["token_to_index"].get(
        c, vocab["token_to_index"][UNK_TOKEN]) for c in raw_question]
    question = question + [vocab["token_to_index"][PAD_TOKEN]] * (16 - len(question))
    assert vocab["token_to_index"][UNK_TOKEN] not in question
    question = np.array(question)

    mask = np.zeros([len(vertex_vectors)])
    for index, region in enumerate(regions):
        label = region["region_attributes"]["formal_key"]
        key_type = region["region_attributes"]["key_type"]
        if target in label and key_type == "value":
            mask[index] = 1

    vertex_vectors = refine_node_features(vertex_vectors)
    if get_raw:
        print("Question:", raw_question)
        return question, vertex_vectors, edge_matrices, mask, raw_nodes
    return question, vertex_vectors, edge_matrices, mask


def process_position_json(data, vocab, get_raw=False):
    vertex_vectors = []
    raw_nodes = []
    #regions = data["attributes"]["_via_img_metadata"]["regions"]
    regions = data["form"]
    for index, region in enumerate(regions):
        vertex, raw_node = get_node_attributes(region, vocab, index, text_len=MAX_LEN_TEXT)
        vertex_vectors.append(vertex)
        raw_nodes.append(raw_node)
    vertex_vectors = np.array(vertex_vectors)
    raw_nodes = np.array(raw_nodes)

    edge_matrices = np.zeros([len(vertex_vectors), len(vertex_vectors), 4])
    for index, vertex in enumerate(vertex_vectors):
        neighbors = find_neighbors(vertex, vertex_vectors)
        for j, n in enumerate(neighbors):
            if n is not None:
                edge_matrices[index, n] = np.eye(4)[j]

    node_index = random.randrange(0, len(vertex_vectors))
    neighbors = edge_matrices[node_index]

    #keywords = ["左", "上", "右", "下"]
    keywords = ["header", "other", "question", "answer"]
    relations = ["right", "left", "top", "bottom"]
    question_templates = ["is this", "which lines are"]
    #random_position = random.randint(0, 3)
    random_key = random.randint(0, len(keywords)-1)
    random_relation = random.randint(0, len(relations)-1)

    #raw_question = "%sの'%s'は?" % (keywords[random_position], raw_nodes[node_index])
    k_question = "{0} {1} a {2}?".format(question_templates[0], raw_nodes[node_index], keywords[random_key])
    r_question = "{0} {1} of this {2}".format(question_templates[1], raw_nodes[node_index], relations[random_relation])
    choose_question = random.choice([(k_question, random_key), (r_question, random_relation)])
    raw_question, random_position = choose_question

    question = [vocab["token_to_index"].get(
        c, vocab["token_to_index"][UNK_TOKEN]) for c in raw_question]
    question = question + [vocab["token_to_index"][PAD_TOKEN]] * max(MAX_LEN_TEXT - len(question), 0)
    question = np.array(question)

    """
    keywords = ["左", "上", "右", "下"]
    random_position = random.randint(0, 3)
    random_position = np.eye(4)[random_position]
    raw_question = "%sの'%s'は?" % (keywords[random_position], raw_nodes[node_index])
    vertex_vectors = encode_bow(raw_nodes, vocab)
    question = vertex_vectors[node_index]
    """

    mask = np.zeros([len(vertex_vectors)])
    for index in range(0, len(raw_nodes)):
        if neighbors[index, random_position] == 1:
            mask[index] = 1

    vertex_vectors = refine_node_features(vertex_vectors)
    if get_raw:
        print("Question:", raw_question)
        return question, vertex_vectors, edge_matrices, mask, raw_nodes
    return question, vertex_vectors, edge_matrices, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_document", required=True, help="scene graph json file")
    parser.add_argument("--vocab_text", required=True, help="vocab file")
    args = parser.parse_args()

    print("Loading data")
    image = cv2.imread("D:/Data/Invoice/hidden_invoice_test/7_0.png")
    with open(args.input_document, encoding="utf-8") as file:
        data = json.load(file)
    vocab = load_vocab(args.vocab_text)

    print("Construct")
    vertex_vectors = []
    regions = data["attributes"]["_via_img_metadata"]["regions"]
    for index, region in enumerate(regions):
        vertex = get_node_attributes(region, vocab, index)
        vertex_vectors.append(vertex)
    vertex_vectors = np.array(vertex_vectors)

    print("Show keys")
    target = ["tel", "fax"]
    keys = set()
    values = {}
    for index, region in enumerate(regions):
        text = region["region_attributes"]["label"]
        label = region["region_attributes"]["formal_key"]
        key_type = region["region_attributes"]["key_type"]
        if any([t in label for t in target]) and key_type == "value":
            values[label] = text
    print(values)

    print("Build edges")
    edge_matrices = {}
    for index, vertex in enumerate(vertex_vectors):
        neighbors = find_neighbors(vertex, vertex_vectors)
        for j, n in enumerate(neighbors):
            if n is not None:
                edge_matrices[(index, n)] = np.eye(4)[j]
    print(vertex_vectors.shape)
    print(len(edge_matrices))
    show_graph(image, vertex_vectors, edge_matrices)


if __name__ == "__main__":
    main()
