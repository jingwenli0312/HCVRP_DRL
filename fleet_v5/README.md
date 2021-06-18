# Heterogeneous Fleet with Five Vehicles

Attention based model for learning to solve the Heterogeneous Capacitated Vehicle Routing Problem (HCVRP) with both min-max and min-sum objective. Training with REINFORCE with greedy rollout baseline.

## Usage

### Generating data

Training data is generated on the fly. To generate test data (same as used in the paper) for 3 vehicles and 40 customers:
```bash
python generate_data.py --veh_num 5 --graph_size 80
```

### Training

For training HCVRP instances with 40 nodes and min-max objective and using rollout as REINFORCE baseline:
```bash
python run.py --graph_size 80 --baseline rollout --run_name 'hcvrp80_rollout' --obj min-max
```
For training HCVRP instances with 40 nodes and min-sum objective and using rollout as REINFORCE baseline:
```bash
python run.py --graph_size 80 --baseline rollout --run_name 'hcvrp80_rollout' --obj min-sum
```

#### Multiple GPUs
By default, training will happen *on all available GPUs*. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=2,3 python run.py 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 nodes).

#### Warm start
You can initialize a run using a pretrained model by using the `--load_path` option:
```bash
python run.py --graph_size 80 --load_path outputs/hcvrp80_rollout/hcvrp80_rollout_{datetime}/epoch-49.pt
```

The `--load_path` option can also be used to load an earlier run, in which case also the optimizer state will be loaded:
```bash
python run.py --graph_size 80 --load_path outputs/hcvrp80_rollout/hcvrp80_rollout_{datetime}/epoch-{num}.pt
```


### Evaluation
To evaluate a model, you can add the `--eval-only` flag to `run.py`, or use `eval.py`, which will additionally measure timing and save the results:
```bash
python eval.py data/hcvrp/hcvrp_80_seed24610.pkl --model outputs/hcvrp80_rollout/hcvrp80_rollout_{datetime}/epoch-{num}.pt --decode_strategy greedy
```
If the epoch is not specified, by default the last one in the folder will be used.

#### Sampling
To report the best of 1280 sampled solutions, use
```bash
python eval.py data/hcvrp/hcvrp_80_seed24610.pkl --model outputs/hcvrp80_rollout/hcvrp80_rollout_{datetime}/epoch-{num}.pt --decode_strategy sample --width 1280 --eval_batch_size 1
```

