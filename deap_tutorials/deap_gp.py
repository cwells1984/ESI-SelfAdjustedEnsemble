# Follows GP tutorial
# https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html
from deap import gp
import operator

# add primitives
pset = gp.PrimitiveSet("main", 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addTerminal(3)

# rename arguments
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    expr = gp.genFull(pset, min_=1, max_=3)
    tree = gp.PrimitiveTree(expr)
    print(tree)