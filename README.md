# x-flux-ip-adapter
The IP-Adapter training scripts and inference for Flux Model, which is implemented based on [X-Lab](https://github.com/XLabs-AI/x-flux).
- Here we provide two method to repeat the training, i.e. `run.py` or `run.sh`
- The training configuration is in `train_configs` folder
- Provide a example inference configuration as `IPAdapter_inference.yaml`

## TODO
- [ ] Providing the final effect image examples
- [ ] Adjusting multi-GPU training with accelerate for WebDataset 
- [ ] Providing the implementation based on diffusers

## Dataset 
Two kinds of dataet is allowed
- WebDataset: Still don't support multi-GPU training (for accelerate reason 🤔 I think, if anyone can help can't be better
- Local dataset: json as followed
  ```python
  [
  {"image_file": "1.png", "text": "A dog"},
  {"image_file": "2.png", "text": "A cat"},
  ...
  ]
  ```

## deepspeed + accelerate config example:
  ```yaml
  compute_environment: LOCAL_MACHINE
  debug: true
  deepspeed_config:
   gradient_accumulation_steps: 2
   gradient_clipping: 1.0
   offload_optimizer_device: none
   offload_param_device: none
   zero3_init_flag: false
   zero_stage: 2
  distributed_type: DEEPSPEED
  downcast_bf16: 'no'
  enable_cpu_affinity: false
  machine_rank: 0
  main_training_function: main
  mixed_precision: bf16
  num_machines: 1
  num_processes: 2
  rdzv_backend: static
  same_network: true
  tpu_env: []
  tpu_use_cluster: false
  tpu_use_sudo: false
  use_cpu: false
  ```
## inference script
```bash
python main.py --config x-flux/IPAdapter_inference.yaml 
```
