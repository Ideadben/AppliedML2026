import numpy as np
from collections import Counter

paths = {
    'kmeans': 'InitialProject/Clustering/Output/Clustering_BenjaminSiddique_KMeans.csv',
    'gmm1': 'InitialProject/Clustering/Output/Clustering_BenjaminSiddique_GMM1.csv',
    'gmm2': 'InitialProject/Clustering/Output/Clustering_BenjaminSiddique_GMM2.csv',
}

def read_labels(path):
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts=line.split(',')
            if len(parts)>=2:
                labels.append(parts[1])
    return labels


def to_prob(counter, all_keys, eps=1e-12):
    # produce probability vector aligned to all_keys
    total = sum(counter.values())
    probs = np.array([ (counter.get(k,0)/total) for k in all_keys ], dtype=float)
    # smooth zeros
    probs = probs + eps
    probs = probs / probs.sum()
    return probs


def kl(p,q):
    return np.sum(p * np.log(p / q))


def js(p,q):
    m = 0.5*(p+q)
    return 0.5*(kl(p,m) + kl(q,m))

labels = {name: read_labels(p) for name,p in paths.items()}

counters = {name: Counter(l) for name,l in labels.items()}

all_keys = sorted({k for c in counters.values() for k in c.keys()}, key=lambda x:int(x))

probs = {name: to_prob(counters[name], all_keys) for name in counters}

pairs = [('kmeans','gmm1'),('kmeans','gmm2'),('gmm1','gmm2')]

print('aligned_label_keys =', all_keys)
print()
for a,b in pairs:
    p = probs[a]
    q = probs[b]
    kl_ab = kl(p,q)
    kl_ba = kl(q,p)
    js_ab = js(p,q)
    print(f'{a} vs {b}: KL({a}||{b})={kl_ab:.6f}, KL({b}||{a})={kl_ba:.6f}, JS={js_ab:.6f}')

# save to CSV
out_lines = ['pair,kl_ab,kl_ba,js']
for a,b in pairs:
    p = probs[a]
    q = probs[b]
    out_lines.append(f'{a}_vs_{b},{kl(p,q):.6f},{kl(q,p):.6f},{js(p,q):.6f}')
with open('InitialProject/Clustering/kl_divergences.csv','w',encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
print('\nSaved results to InitialProject/Clustering/kl_divergences.csv')
