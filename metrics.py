def dice(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 0
    else:
        return 2 * len(set(set1).intersection(set(set2))) / (len(set(set1)) + len(set(set2)))



def jaccard(set1,set2):
    return len(set(set1).intersection(set(set2)))/len(set(set1).union(set(set2)))
