import glob
import random
import itertools
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import csv
import os

def make_partition(
    signers: List[int],
    pair_genuine_genuine: List[Tuple[int, int]],
    pair_genuine_forged: List[Tuple[int, int]],
):
    samples = []
    for signer_id in signers:
        sub_pair_genuine_forged = random.sample(pair_genuine_forged, len(pair_genuine_genuine))
        genuine_genuine = list(itertools.zip_longest(pair_genuine_genuine, [], fillvalue=1)) # y = 1
        genuine_genuine = list(map(lambda sample: (signer_id, *sample[0], sample[1]), genuine_genuine))
        samples.extend(genuine_genuine)
        genuine_forged = list(itertools.zip_longest(sub_pair_genuine_forged, [], fillvalue=0)) # y = 0
        genuine_forged = list(map(lambda sample: (signer_id, *sample[0], sample[1]), genuine_forged))
        samples.extend(genuine_forged)
    return samples

def write_csv(file_path, samples):
    with open(file_path, 'wt') as f:
        writer = csv.writer(f)
        writer.writerows(samples)

def prepare_SVC2004(M: int, K: int, random_state=0, data_dir='data/SVC2004'):
    def get_path(row):
        writer_id, x1, x2, y = row
        if y == 1:
            x1 = os.path.join(data_dir, f'user{writer_id}', 'genuine', f'U{writer_id}S{x1}.jpg')
            x2 = os.path.join(data_dir, f'user{writer_id}', 'genuine', f'U{writer_id}S{x2}.jpg')
        else:
            x1 = os.path.join(data_dir, f'user{writer_id}', 'genuine', f'U{writer_id}S{x1}.jpg')
            x2 = os.path.join(data_dir, f'user{writer_id}', 'forged', f'U{writer_id}S{20+x2}.jpg')
        return x1, x2, y # drop writer_id

    random.seed(random_state)
    signers = list(range(1, K+1))
    num_genuine_sign = 20
    num_forged_sign = 20

    train_signers, test_signers = train_test_split(signers, test_size=K-M)
    pair_genuine_genuine = list(itertools.combinations(range(1, num_genuine_sign+1), 2))
    pair_genuine_forged = list(itertools.product(range(1, num_genuine_sign+1), range(1, num_forged_sign+1)))

    train_samples = make_partition(train_signers, pair_genuine_genuine, pair_genuine_forged)
    train_samples = list(map(get_path, train_samples))
    write_csv(os.path.join(data_dir, 'train.csv'), train_samples)
    test_samples = make_partition(test_signers, pair_genuine_genuine, pair_genuine_forged)
    test_samples = list(map(get_path, test_samples))
    write_csv(os.path.join(data_dir, 'test.csv'), test_samples)

def prepare_CEDAR(M: int, K: int, random_state=0, data_dir='data/CEDAR'):
    def get_path(row):
        writer_id, x1, x2, y = row
        if y == 1:
            x1 = os.path.join(data_dir, 'full_org', f'original_{writer_id}_{x1}.png')
            x2 = os.path.join(data_dir, 'full_org', f'original_{writer_id}_{x2}.png')
        else:
            x1 = os.path.join(data_dir, 'full_org', f'original_{writer_id}_{x1}.png')
            x2 = os.path.join(data_dir, 'full_forg', f'forgeries_{writer_id}_{x2}.png')
        return x1, x2, y # drop writer_id

    random.seed(random_state)
    signers = list(range(1, K+1))
    num_genuine_sign = 24
    num_forged_sign = 24

    train_signers, test_signers = train_test_split(signers, test_size=K-M)
    pair_genuine_genuine = list(itertools.combinations(range(1, num_genuine_sign+1), 2))
    pair_genuine_forged = list(itertools.product(range(1, num_genuine_sign+1), range(1, num_forged_sign+1)))

    train_samples = make_partition(train_signers, pair_genuine_genuine, pair_genuine_forged)
    train_samples = list(map(get_path, train_samples))
    write_csv(os.path.join(data_dir, 'train.csv'), train_samples)
    test_samples = make_partition(test_signers, pair_genuine_genuine, pair_genuine_forged)
    test_samples = list(map(get_path, test_samples))
    write_csv(os.path.join(data_dir, 'test.csv'), test_samples)


def prepare_BHSig260(M: int, K: int, random_state=0, data_dir='data/BHSig260/Bengali'):
    def get_path(row):
        writer_id, x1, x2, y = row
        if y == 1:
            x1 = os.path.join(data_dir, f'{writer_id:03d}', f'B-S-{writer_id}-G-{x1:02d}.tif')
            x2 = os.path.join(data_dir, f'{writer_id:03d}', f'B-S-{writer_id}-G-{x2:02d}.tif')
        else:
            x1 = os.path.join(data_dir, f'{writer_id:03d}', f'B-S-{writer_id}-G-{x1:02d}.tif')
            x2 = os.path.join(data_dir, f'{writer_id:03d}', f'B-S-{writer_id}-F-{x2:02d}.tif')
        return x1, x2, y # drop writer_id

    random.seed(random_state)
    signers = list(range(1, K+1))
    num_genuine_sign = 24
    num_forged_sign = 30

    train_signers, test_signers = train_test_split(signers, test_size=K-M)
    pair_genuine_genuine = list(itertools.combinations(range(1, num_genuine_sign+1), 2))
    pair_genuine_forged = list(itertools.product(range(1, num_genuine_sign+1), range(1, num_forged_sign+1)))

    train_samples = make_partition(train_signers, pair_genuine_genuine, pair_genuine_forged)
    train_samples = list(map(get_path, train_samples))
    write_csv(os.path.join(data_dir, 'train.csv'), train_samples)
    test_samples = make_partition(test_signers, pair_genuine_genuine, pair_genuine_forged)
    test_samples = list(map(get_path, test_samples))
    write_csv(os.path.join(data_dir, 'test.csv'), test_samples)

if __name__ == "__main__":
    print('Preparing CEDAR dataset..')
    #prepare_CEDAR(M=50, K=55)
    prepare_SVC2004(M=32, K=40)
    # print('Preparing Bengali dataset..')
    # prepare_BHSig260(M=50, K=100, data_dir='data/BHSig260/Bengali')
    # print('Preparing Hindi dataset..')
    # prepare_BHSig260(M=100, K=160, data_dir='data/BHSig260/Hindi')
    print('Done')