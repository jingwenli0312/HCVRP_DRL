import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
import torch
import pickle
import argparse

def generate_hcvrp_data(dataset_size, hcvrp_size, veh_num):
    data = []
    for seed in range(24601, 24611):
        rnd = np.random.RandomState(seed)

        loc = rnd.uniform(0, 1, size=(dataset_size, hcvrp_size + 1, 2))
        depot = loc[:, -1]
        cust = loc[:, :-1]
        d = rnd.randint(1, 10, [dataset_size, hcvrp_size+1])
        d = d[:, :-1]  # the demand of depot is 0, which do not need to generate here

        if veh_num == 3:
            cap = [20., 25., 30.]
            thedata = list(zip(depot.tolist(),  # Depot location
                               cust.tolist(),
                               d.tolist(),
                               np.full((dataset_size, 3), cap).tolist()
                                ))
            data.append(thedata)

        elif veh_num == 5:
            cap = [20., 25., 30., 35., 40.]
            thedata = list(zip(depot.tolist(),  # Depot location
                               cust.tolist(),
                               d.tolist(),
                               np.full((dataset_size, 5), cap).tolist()
                               ))
            data.append(thedata)

    data = np.array(data).reshape(1280, 4)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--dataset_size", type=int, default=128, help="1/10 Size of the dataset")
    parser.add_argument("--veh_num", type=int, default=3, help="number of the vehicles; 3 or 5")
    parser.add_argument('--graph_size', type=int, default=40,
                        help="Sizes of problem instances: {40, 60, 80, 100, 120} for 3 vehicles, "
                             "{80, 100, 120, 140, 160} for 5 vehicles")

    opts = parser.parse_args()
    data_dir = 'data'
    problem = 'hcvrp'
    datadir = os.path.join(data_dir, problem)
    os.makedirs(datadir, exist_ok=True)
    seed = 24610  # the last seed used for generating HCVRP data
    np.random.seed(seed)
    print(opts.dataset_size, opts.graph_size)
    filename = os.path.join(datadir, '{}_v{}_{}_seed{}.pkl'.format(problem, opts.veh_num, opts.graph_size, seed))

    dataset = generate_hcvrp_data(opts.dataset_size, opts.graph_size, opts.veh_num)
    print(dataset[0])
    save_dataset(dataset, filename)



