import numpy as np
import random as rd
from inspect import signature
from abc import abstractmethod

class Forest():

    def __init__(self, non_terminals, constants, n_variables, p_terminal, max_depth) -> None:
        self.p_terminal = p_terminal
        self.n_terminals = len(constants) + n_variables
        self.n_constants = len(constants)
        self.n_variables = n_variables
        self.n_internals = len(non_terminals)
        self.s_internals = [t.__name__ for t in non_terminals]
        self.max_depth = max_depth

        self.arities = [len(signature(t).parameters) for t in non_terminals]
        self.internal = [np.frompyfunc(non_terminals[i], self.arities[i], 1) for i in range(len(non_terminals))]
        self.constants = constants

        self.trees = []
        self.scores = []

    def evolve(self, n_trees, n_iterations, p_crossover, p_mutation, score_function, X, y=None):
        self.n_trees = n_trees
        self.init_trees()
        for iteration in range(n_iterations):
            results = [t.evaluate(X) for t in self.trees]
            if y is None: self.scores = [score_function(r) for r in results]
            else: self.scores = [score_function(r,y) for r in results]
            self.grow_new_trees()
            self.crossover_trees(p_crossover)
            self.mutate_trees(p_mutation)
            print('iteration={}\tmin={}\tmax={}'.format(iteration, np.min(self.scores), np.max(self.scores)))

    def init_trees(self):
        self.trees = [self.build_tree(self.max_depth, None, t) for t in range(self.n_trees)]

    def build_tree(self, depth, parent, branch_index):
        if rd.random() < self.p_terminal or depth == 1:
            index = rd.randrange(self.n_terminals)
            if index < self.n_constants:
                tree = ConstantNode(depth-1, parent, branch_index, self.constants[index])
            else:
                tree = VariableNode(depth-1, parent, branch_index, index-self.n_constants)
        else:
            tree = TreeNode(depth-1, parent, branch_index, self, rd.randrange(self.n_internals))
        return tree

    def grow_new_trees(self):
        weights = [score / np.sum(np.absolute(self.scores)) for score in np.absolute(self.scores)]
        new_trees = rd.choices(self.trees, weights, k=self.n_trees)

    def crossover_trees(self, p_crossover):
        for pair in range(int(self.n_trees/2)):
            if rd.random() < p_crossover:
                tree_0 = self.trees[2*pair+0]
                tree_1 = self.trees[2*pair+1]
                k0 = 1 + rd.randrange(tree_0.set_size())
                k1 = 1 + rd.randrange(tree_1.set_size())
                (parent_0, branch_index_0) = tree_0.get_subtree_detaills(k0)
                (parent_1, branch_index_1) = tree_1.get_subtree_detaills(k1)
                if parent_0 is not None and parent_1 is not None:
                    subtree_0 = parent_0.children[branch_index_0]
                    subtree_1 = parent_1.children[branch_index_1]
                    size_difference0 = subtree_1.size - subtree_0.size
                    size_difference1 = subtree_0.size - subtree_1.size
                    parent_0.children[branch_index_0] = subtree_1
                    parent_1.children[branch_index_1] = subtree_0
                    parent_0.update_parent_size(size_difference0)
                    parent_1.update_parent_size(size_difference1)

    def mutate_trees(self, p_mutation):
        for t in self.trees:
            t.set_size()
            if rd.random() < p_mutation:
                k = 1+rd.randrange(t.size)
                (parent, branch_index) = t.get_subtree_detaills(k)
                if parent is None:
                    self.trees[branch_index] = self.build_tree(self.max_depth, parent, branch_index)
                else:
                    old_subtree = parent.children[branch_index]
                    new_subtree = self.build_tree(old_subtree.depth+1, parent, branch_index)
                    new_subtree.set_size()
                    size_difference = new_subtree.size-old_subtree.size
                    parent.children[branch_index] = new_subtree
                    parent.update_parent_size(size_difference)

    def as_string(self):
        return "\n".join([t.as_string() for t in self.trees])

    def get_best_tree(self):
        return self.trees[np.argmax(self.scores)]

class Node():

    def __init__(self, depth, parent, branch_index):
        self.depth = depth
        self.parent = parent
        self.branch_index = branch_index

    @abstractmethod
    def evaluate(self, X):
        pass

    @abstractmethod
    def as_string(self):
        pass

    @abstractmethod
    def set_size(self):
        pass

    @abstractmethod
    def get_subtree_detaills(k):
        pass

    def update_parent_size(self, size_difference):
        self.size += size_difference
        if self.parent is not None:
            self.parent.update_parent_size(size_difference)

class TreeNode(Node):

    def __init__(self, depth, parent, branch_index, forest, index) -> None:
        super().__init__(depth, parent, branch_index)
        self.parent = parent
        self.idx = index
        self.arity = forest.arities[index]
        self.function = forest.internal[index]
        self.name = forest.s_internals[index]
        self.children = [None]*self.arity

        for i in range(self.arity):
            if rd.random() < forest.p_terminal or depth == 1:
                index = rd.randrange(forest.n_terminals)
                if index < forest.n_constants:
                    self.children[i] = ConstantNode(depth-1, self, i, forest.constants[index]) 
                else:
                    self.children[i] = VariableNode(depth-1, self, i, index-forest.n_constants)
            else:
                self.children[i] = TreeNode(depth-1, self, i, forest, rd.randrange(forest.n_internals))

    def evaluate(self, X):
        child_values = [c.evaluate(X) for c in self.children]
        return self.function(*child_values)

    def as_string(self):
        return self.name + "(" + ", ".join([c.as_string() for c in self.children]) + ")"

    def set_size(self):
        self.size = 1 + max([c.set_size() for c in self.children])
        return self.size

    def get_subtree_detaills(self, k):
        if k == 1:
            return (self.parent, self.branch_index)
        else:
            k -= 1
            for c in self.children:
                if k <= c.size:
                    return c.get_subtree_detaills(k)
                else:
                    k -= c.size

class ConstantNode(Node):

    def __init__(self, depth, parent, branch_index, constant) -> None:
        super().__init__(depth, parent, branch_index)
        self.constant = constant

    def evaluate(self, X):
        result = np.empty(len(X))
        result.fill(self.constant)
        return result

    def as_string(self):
        return str(self.constant)

    def set_size(self):
        self.size = 1
        return self.size

    def get_subtree_detaills(self, k):
        return (self.parent, self.branch_index)

class VariableNode(Node):

    def __init__(self, depth, parent, branch_index, var_index) -> None:
        super().__init__(depth, parent, branch_index)
        self.var_index = var_index

    def evaluate(self, X):
        return X[:, self.var_index]

    def as_string(self):
        return "x"+str(self.var_index)

    def set_size(self):
        self.size = 1
        return self.size
    
    def get_subtree_detaills(self, k):
        return (self.parent, self.branch_index)
