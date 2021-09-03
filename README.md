# Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem

Attention based model for learning to solve the Heterogeneous Capacitated Vehicle Routing Problem (HCVRP) with both min-max and min-sum objective. Training with REINFORCE with greedy rollout baseline.

## Paper
For more details, please see our paper [Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem](https://github.com/Demon0312/HCVRP_DRL/blob/main/paper/paper.pdf) which has been accepted at [IEEE Transactions on Cybernetics]. If this code is useful for your work, please cite our paper:

```
@article{li2021hcvrp,
  title={Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem},
  author={Li, Jingwen and Ma, Yining and Gao, Ruize and Cao, Zhiguang and Andrew, Lim and Song, Wen and Zhang, Jie},
  journal={IEEE Transactions on Cybernetics},
  year={2021}
}
``` 

## Dependencies

* Python>=3.6
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)=0.4
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)

## Details
For more details, please see the fleet_v3 and fleet_v5 for HCVRP with three vehicles and five vehicles, respectively.

