import numpy as np
from collections import namedtuple, Counter
import nltk
from nltk.corpus import dependency_treebank
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx
from tqdm import tqdm
Arc = namedtuple('Arc', ['head', 'tail', 'weight'])
ROOT_NODE = {'word': 'ROOT', 'tag':'ROOT','address':0}

class MSTParser:
    def __init__(self, train_set):
        self.train_set = train_set
        self.words = Counter()
        self.tags = Counter()
        self.weights = Counter()

    def get_feature_vec(self, dependency_graph):
        nodes = dependency_graph.nodes
        root_index = dependency_graph.root['address']
        self.words[('ROOT', nodes[root_index]['word'])] += 1
        self.tags[('ROOT', nodes[root_index]['tag'])] += 1

        for i in range(1, len(nodes)):
            dest_node = dependency_graph.nodes[i]
            source_node = dependency_graph.nodes[dest_node['head']]
            self.words[(source_node['word'], dest_node['word'])] += 1
            self.tags[(source_node['tag'], dest_node['tag'])] += 1

        return self.words | self.tags



    def get_sentence_graph_2(self, sentence, feature_func, weights):
        arcs = []
        nodes = sentence.nodes
        for i, node in nodes.items():
            if i == 0:
                continue
            score = 0
            u_ind = node['head']
            v_ind = node['address']
            u_word = nodes[u_ind]['word'] if u_ind else 'ROOT'
            u_tag = nodes[u_ind]['tag'] if u_ind else 'ROOT'
            v_word = nodes[v_ind]['word']
            v_tag = nodes[v_ind]['tag']
            word_arc = (u_word, v_word)
            tag_arc = (u_tag, v_tag)
            for arc in [word_arc, tag_arc]:
                score -= feature_func[arc] * weights[arc]
            arcs.append(Arc(head=u_ind, tail=v_ind, weight=score))
        return arcs


    def get_sentence_graph(self, sentence, feature_func, weights):
        arcs = []
        nodes = sentence.nodes
        for i, node1 in nodes.items():
            for j, node2 in nodes.items():
                if i == 0 or j==0 or i==j:
                    continue
                score = 0
                u_ind = node1['address']
                v_ind = node2['address']
                u_word = nodes[u_ind]['word'] if u_ind else 'ROOT'
                u_tag = nodes[u_ind]['tag'] if u_ind else 'ROOT'
                v_word = nodes[v_ind]['word']
                v_tag = nodes[v_ind]['tag']
                word_arc = (u_word, v_word)
                tag_arc = (u_tag, v_tag)
                for arc in [word_arc, tag_arc]:
                    score -= feature_func[arc] * weights[arc]
                arcs.append(Arc(head=u_ind, tail=v_ind, weight=score))
        return arcs
    def calc_sentence_mst(self, sentence, feature_func, weights):
        sentence_score = self.get_sentence_graph(sentence, feature_func, weights)
        return min_spanning_arborescence_nx(sentence_score, 0)


    def calc_mst_feature_vec(self, mst, sentence):
        features = Counter()
        nodes = sentence.nodes
        for arc in mst.values():
            u_ind = arc.head
            v_ind = arc.tail
            if u_ind == 0:
                features[(nodes[u_ind]['word'], nodes[v_ind]['word'])] += 1
                features[(nodes[u_ind]['tag'], nodes[v_ind]['tag'])] += 1
            else:
                features[('ROOT', nodes[v_ind]['word'])] += 1
                features[('ROOT', nodes[v_ind]['tag'])] += 1
        return features

    def perceptron(self, iterations=2, lr=1):
        N = len(self.train_set) * iterations
        sentences_range = np.arange(len(self.train_set))
        for iter in range(iterations):
            np.random.shuffle(sentences_range)
            for i in tqdm(sentences_range, f'iteration {iter}'):
                sentence = self.train_set[i]
                sen_feature_vec = self.get_feature_vec(sentence)
                mst = self.calc_sentence_mst(sentence, sen_feature_vec, self.weights)
                mst_feature_vec = self.calc_mst_feature_vec(mst, sentence)
                weight_diff = sen_feature_vec - mst_feature_vec
                for key, val in weight_diff.items():
                    weight_diff[key] = val * lr
                self.weights += weight_diff

        for key in self.weights:
            self.weights[key] /= N

    def evaluate(self, test_set):
        acc = 0
        for sentence in test_set:
            feature_func = self.get_feature_vec(sentence)
            mst = self.calc_sentence_mst(sentence, feature_func, self.weights)
            mst_arcs = set([(arc.head, arc.tail) for arc in mst.values()])
            sentence_arcs = set([(node['head'], node['address']) for node in sentence.nodes.values() if node['head']])
            intersected_arcs = sentence_arcs.intersection(mst_arcs)
            acc += len(intersected_arcs) / len(sentence.nodes)
        return acc / len(test_set)


def main():
    sentences = dependency_treebank.parsed_sents()
    train_set, test_set = sentences[:int(len(sentences) * 0.9)], \
                          sentences[int(len(sentences) * 0.9):]
    parser = MSTParser(train_set)
    parser.perceptron()
    print(parser.evaluate(test_set))



if __name__ == '__main__':
    main()
