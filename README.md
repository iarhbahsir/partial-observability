# partial-observability

### Abstract:
In this work we explore how well modern actor-critic methods designed for learning in fully observable environments generalize to learning under partial observability. We consider different actor-critic algorithms, as well as different forms of partial observability and environment complexity that the agent may be presented with. Under the replicated noisy and faulty sensor conditions, results show greater difficulty when learning with noisy observations than faulty ones. We observe that, in general, SAC is more resistant to noisy conditions while TD3 is better able to perform under faulty conditions. In addition, the dimensionality of the environment appears to be inversely related to the difficulty of learning under partial observability.

### Video available here:
https://youtu.be/1SPu-Azoqg0

### To run:

From inside repo directory: ```python script.py environment_name run_name_prefix pomdp_type```

script.py from ```[sac.py, td3.py]```, environment name from ```[Hopper-v2, Ant-v2]```, run_name_prefix arbitrary, and pomdp_type from ```[noisy, faulty, fully-obs]```

### Note:
Earlier commits in the development of the TD3 and SAC implementations can be found in this repository: https://github.com/iarhbahsir/rl-algorithms