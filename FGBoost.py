from math import exp
from TILDE import TILDE
from random import sample
from copy import deepcopy

def sigmoid(x):
    """returns e^x/(1+e^x)
    """

    return (exp(x)/(1+exp(x)))

class GBoost(object):


    def __init__(self,data,pos,neg,bk,target,max_depth=2):
        """classification class to predict target
           conditioned on data and background
           using functional gradient ascent on sigmoid
           probability model assumption of target given data
        """

        self.data = data
        self.examples = {}
        self.pos = pos
        self.neg = neg
        #initial model assumed P(target|data)=0.5 to target=1,0
        for ex in pos+neg:
            self.examples[ex] = 0
        self.bk = bk
        self.target = target
        self.max_depth = max_depth
        self.boosted_trees = []

    def learn_MAP(self,knowledge,k=10):
        """learns set of k boosted trees
        """
        
        gradients = {}
        for i in range(k):

            #create TILDE(R) tree object
            tree_i = TILDE(typ="regression",score="WV",max_depth=self.max_depth)

            #subsample negatives if too many for each tree
            sampled_neg = deepcopy(self.neg)
            if len(self.neg) > 2*len(self.pos):
                sampled_neg = sample(self.neg,2*len(self.pos))

            #compute gradients as I-P
            for ex in self.examples:
                parameter = knowledge.calculate_parameter(self.data,
                                                          ex,
                                                          self.examples[ex])
                p = sigmoid(self.examples[ex])
                if ex in self.pos:
                    gradients[ex] = 1-p - parameter
                elif ex in sampled_neg:
                    gradients[ex] = 0-p - parameter

            #fit tree on gradients
            tree_i.learn(self.data,self.bk,self.target,examples=gradients)
            
            #recompute example values as previous example value + tree_i value
            for ex in self.examples:
                tree_i_value = tree_i.infer(self.data,ex)
                self.examples[ex] += tree_i_value

            #add tree to boosted_trees
            self.boosted_trees.append(tree_i)

    def learn(self,k=10):
        """learns set of k boosted trees
        """
        
        gradients = {}
        for i in range(k):

            #create TILDE(R) tree object
            tree_i = TILDE(typ="regression",score="WV",max_depth=self.max_depth)

            #subsample negatives if too many for each tree
            sampled_neg = deepcopy(self.neg)
            if len(self.neg) > 2*len(self.pos):
                sampled_neg = sample(self.neg,2*len(self.pos))

            #compute gradients as I-P
            for ex in self.examples:
                p = sigmoid(self.examples[ex])
                if ex in self.pos:
                    gradients[ex] = 1-p
                elif ex in sampled_neg:
                    gradients[ex] = 0-p

            #fit tree on gradients
            tree_i.learn(self.data,self.bk,self.target,examples=gradients)
            
            #recompute example values as previous example value + tree_i value
            for ex in self.examples:
                tree_i_value = tree_i.infer(self.data,ex)
                self.examples[ex] += 0.01*tree_i_value

            #add tree to boosted_trees
            self.boosted_trees.append(tree_i)

    def infer(self,data,examples,k=10):
        """infer value of examples from data
           and a subset or all of the trees
        """

        example_values = []
        for example in examples:
            example_value = 0
            for i in range(k):
                tree_i = self.boosted_trees[i]
                tree_i_value = tree_i.infer(data,example)
                example_value += tree_i_value
            example_values.append(sigmoid(example_value))

        return example_values

class MetricBoost(object):


    def __init__(self,data,pos,neg,bk,prior,target,max_depth=2):
        """classification class to predict target
           conditioned on data and background
           using functional gradient ascent on sigmoid
           probability model assumption of target given data
        """

        self.data = data
        self.examples = {}
        self.pos = pos
        self.neg = neg
        #initial model assumed P(target|data)=0.5 to target=1,0
        for ex in pos+neg:
            self.examples[ex] = prior[ex]
        self.bk = bk
        self.target = target
        self.max_depth = max_depth
        self.boosted_trees = []

    def compute_gradient(self,ex,pos,neg,examples):
        """computes gradient for all examples
           according to LMNN objective
        """

        k = 2
        imposters = []

        distances = {}
        for other_ex in examples:

            #skip same example
            if ex == other_ex:
                continue

            #add other class to imposters
            if other_ex in pos and ex in neg:
                imposters.append(other_ex)
                continue
            if other_ex in neg and ex in pos:
                imposters.append(other_ex)
                continue

            #if same class, calculate distance
            distances[other_ex] = (examples[ex]-examples[other_ex])**2

        #get k nearest same class neighbors
        k_nearest_distances = sorted(list(distances.values()))[0:k]
        nearest_neighbors = [n for n in distances if distances[n] in k_nearest_distances]
        
        #calculate gradient for first loss component
        first_term = 0.0
        for neighbor in nearest_neighbors:
            first_term += -2*(examples[ex]-examples[neighbor])*examples[neighbor]
            
        #calculate gradient for second loss component
        second_term = 0.0
        for neighbor in nearest_neighbors:
            for imposter in imposters:
                if (((examples[ex]-examples[neighbor])**2) - ((examples[ex]-examples[imposter])**2) + 1 > 0):
                    second_term += -2*(examples[ex]-examples[neighbor])*examples[neighbor]
                    second_term += 2*(examples[ex]-examples[imposter])*examples[imposter]

        return (first_term+second_term)

    def learn(self,k=10):
        """learns set of k boosted trees
        """
        
        gradients = {}
        for i in range(k):

            #create TILDE(R) tree object
            tree_i = TILDE(typ="regression",score="WV",max_depth=self.max_depth)

            #subsample negatives if too many for each tree
            sampled_neg = deepcopy(self.neg)
            if len(self.neg) > 2*len(self.pos):
                sampled_neg = sample(self.neg,2*len(self.pos))

            #compute gradients using LMNN loss function
            for ex in self.examples:
                gradient = self.compute_gradient(ex,
                                                 self.pos,
                                                 self.neg,
                                                 self.examples)
                gradients[ex] = gradient


            #fit tree on gradients
            tree_i.learn(self.data,self.bk,self.target,examples=gradients)
            
            #recompute example values as previous example value - gamma*tree_i value
            for ex in self.examples:
                tree_i_value = tree_i.infer(self.data,ex)
                self.examples[ex] -= 0.01*tree_i_value #learning rate
                
            #add tree to boosted_trees
            self.boosted_trees.append(tree_i)

    def infer(self,data,examples,k=10):
        """infer value of examples from data
           and a subset or all of the trees
        """

        example_values = []
        for example in examples:
            example_value = 0
            for i in range(k):
                tree_i = self.boosted_trees[i]
                tree_i_value = tree_i.infer(data,example)
                example_value += tree_i_value
            example_values.append(example_value)

        return example_values
