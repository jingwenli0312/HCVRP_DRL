# Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem

Attention based model for learning to solve the Heterogeneous Capacitated Vehicle Routing Problem (HCVRP) with both min-max and min-sum objective. Training with REINFORCE with greedy rollout baseline.

## Paper
For more details, please see our paper： Jingwen Li, Yining Ma, Ruize Gao, Zhiguang Cao, Andrew Lim, Wen Song, Jie Zhang. [Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem](https://ieeexplore.ieee.org/document/9547060). IEEE Transactions on Cybernetics, 2021. If this code is useful for your work, please cite our paper,

```
@article{li2021hcvrp,
  title={Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem},
  author={Li, Jingwen and Ma, Yining and Gao, Ruize and Cao, Zhiguang and Andrew, Lim and Song, Wen and Zhang, Jie},
  journal={IEEE Transactions on Cybernetics},
  volume={52},
  number={12},
  pages={13572--13585},
  year={2022},
  publisher={IEEE}，
  doi={10.1109/TCYB.2021.3111082}
}
``` 

## Dependencies

* Python>=3.7
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)=1.3.0
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)

## Details
For more details, please see the fleet_v3 and fleet_v5 for HCVRP with three vehicles and five vehicles, respectively.

