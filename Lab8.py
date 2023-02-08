#Tom Teasdale
#Project - Lab 8 - Bayesian Inference

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

"Return a set containing the ancestors of var"
def get_ancestors(net, var):
    ancestors = set()
    queue = [var]
    while queue:
        node = queue.pop(0)
        for parent in net.get_parents(node):
            if parent not in ancestors:
                ancestors.add(parent)
                queue.append(parent)
    return ancestors


"Returns a set containing the descendants of var"
def get_descendants(net, var):
    descendants = set()
    queue = [var]
    while queue:
        node = queue.pop(0)
        for child in net.get_children(node):
            if child not in descendants:
                descendants.add(child)
                queue.append(child)
    return descendants


"Returns a set containing the non-descendants of var"    
def get_nondescendants(net, var):
    descendants = get_descendants(net, var)
    descendants.add(var)
    return set(net.get_variables()) - descendants

#### Part 2: Computing Probability #############################################

"""
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
"""
def simplify_givens(net, var, givens):
    parents = set(net.get_parents(var))
    descendants = set(net.get_children(var))
    givens_set = set(givens.keys())
    if parents.issubset(givens_set) and not givens_set.intersection(descendants):
        new_givens = givens.copy()
        for descendant in descendants:
            new_givens.pop(descendant, None)
        return new_givens
    else:
        return givens
    
    "Looks up a probability in the Bayes net, or raises LookupError"
def probability_lookup(net, hypothesis, givens=None):
    try:
        return net.get_probability(hypothesis, parents_vals=givens)
    except ValueError:
        try:
            givens = simplify_givens(net, list(hypothesis.keys())[0], givens)
            return net.get_probability(hypothesis, parents_vals=givens)
        except ValueError:
            raise LookupError("The hypothesis you provided contains multiple variables or does not contain the exact parents of the hypothesis' variable.")


    "Uses the chain rule to compute a joint probability"
def probability_joint(net, hypothesis):
    joint_list = []
    for var in hypothesis:
        givens = dict(hypothesis)
        for i in hypothesis:
            if i == var or i not in net.get_parents(var):
                del givens[i]
        joint_list.append(probability_lookup(net, {var: hypothesis[var]}, givens))
    return product(joint_list)

    "Computes a marginal probability as a sum of joint probabilities"    
def probability_marginal(net, hypothesis):
    marg_list = []
    for var in net.combinations(net.get_variables(), hypothesis):
        marg_list.append(probability_joint(net, var))
    return sum(marg_list)

    "Computes a conditional probability as a ratio of marginal probabilities"
def probability_conditional(net, hypothesis, givens=None):
    if not givens:
        return probability_marginal(net, hypothesis)
    for i in hypothesis:
        for j in givens:
            if i == j:
                if hypothesis[i] != givens[j]:
                    return 0.0
                else:
                    return 1.0
    total_dict = dict(hypothesis, **givens)
    return probability_marginal(net, total_dict)/probability_marginal(net, givens)


    "Calls previous functions to compute any probability"
def probability(net, hypothesis, givens=None):
    if not givens:
        return probability_conditional(net, hypothesis)
    else:
        return probability_conditional(net, hypothesis, givens)
