import forest
import math
import numpy as np
import random as rd

rd.seed(12345)

def f (a, b, c):
    return a*b-c

def g (a,b):
    return (a+b)+(a-b)

def h (a, b, c, d):
    return (a+b) - (c-d)

def score_function(x, y):
    res =  np.sum(1/((x-y)**2 + 1e-80)) 
    return res

def build_forest():

    # arguments for the constructor of the forest object
    non_terminals = [f, g, h, math.cos, math.sin]
    constants = [12,34, 0, 1]
    n_variables = 3
    p_terminal = 0.1
    max_depth = 4

    # arguments for the evolution process
    n_trees = 10
    n_iterations = 10
    p_crossover = 0.9
    p_mutation = 0.1
    X = np.array([[1,2,3],[4,5,6],[7,8,9]]).astype('O')
    y = np.array([[6],[14],[24]]).astype('O')

    fo = forest.Forest(non_terminals, constants, n_variables, p_terminal, max_depth)
    fo.evolve(n_trees, n_iterations, p_crossover, p_mutation, score_function, X, y)

    # give out detaills of the found result
    best_tree = fo.get_best_tree()
    print("Best tree: {}\tscore: {}\tpredictions: {}".format(best_tree.as_string(), score_function(best_tree.evaluate(X), y), best_tree.evaluate(X)))

if __name__ == '__main__':
    build_forest()