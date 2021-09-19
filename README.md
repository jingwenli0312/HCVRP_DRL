# Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem

Attention based model for learning to solve the Heterogeneous Capacitated Vehicle Routing Problem (HCVRP) with both min-max and min-sum objective. Training with REINFORCE with greedy rollout baseline.

## Paper
For more details, please see our paper： Jingwen Li, Yining Ma, Ruize Gao, Zhiguang Cao, Andrew Lim, Wen Song, Jie Zhang. [Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem](https://www.researchgate.net/publication/354354946_Deep_Reinforcement_Learning_for_Solving_the_Heterogeneous_Capacitated_Vehicle_Routing_Problem?utm_source=twitter&rgutm_meta1=eHNsLW92c0hUMkM3QWFMS2hDbTBtTWdOdTJpbGRIeVZmWm5CWjJEQ1JXTStkUXZaN0JVUi9rZ01NU2dtbFVJelRXUVd5RjkvNXhWNWl5VFdGVXpyNEtDM3FyOD0%3D). IEEE Transactions on Cybernetics, 2021. If this code is useful for your work, please cite our paper,

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

