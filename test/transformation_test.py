from augment.transformation import Transformation
import copy

if __name__ == '__main__':
    trs = Transformation()
    trss = [trs]
    print trs.get_paras()
    mutates = trs.mutate(trss, 1)
    print "======================== mutate ==========================="
    for i in range(len(mutates)):
        print mutates[i].get_paras()

    print "======================== crossover ==========================="
    trs2 = Transformation(1, 1, 1, 1, 1, 1, 1)
    trs3 = Transformation(1, 1, 0, 1, 0, 0, 1)

    trss = [trs2, trs3]
    crossovers = trs.crossover(trs2, trss, 1)
    for i in range(len(crossovers)):
        print crossovers[i].get_paras()

    print "======================== print parameters ==========================="
    trs_new = Transformation(*(trs.get_paras()))
    print trs_new.get_paras()

    trs4 = Transformation(1, 1, 1, 1, 1, 1, 1)
    print trs4 in trss

    print "======================== object copy ==========================="
    trs5 = copy.copy(trs4)
    trs4.rotation = 10
    print trs4.get_paras()
    print trs5.get_paras()
