# dl_assignment_ed24s401

wandb report link : https://wandb.ai/ed24s401-indian-institute-of-technology-madras/q4_sweep_project/reports/DA6401-Assignment-1--VmlldzoxMTgzOTAwMw
github link: https://github.com/MK-Sindhu/dl_assignment_ed24s401

run configurations : 

```
python train.py --wandb_entity myname --wandb_project myprojectname
```

### Arguments to be supported

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

----------------------------------------------------------------------------------<br>
in order to run a sweep, run the following command:<br>
wandb sweep config.yaml
<br>

configs supported:<br>
```yaml

program: "train.py"
name: "q4_sweep_project"
method: "bayes"
metric:
  goal: maximize
  name: validation_accuracy
parameters:
  epochs:
    values: [5, 10]
  batch_size:
    values: [16, 32, 64]
  num_layers:
    values: [3,4,5]
  hidden_size:
    values: [32, 64, 128]
  weight_decay:
    max: 0.00001 
    min: 0.000001
    distribution: log_uniform_values
  learning_rate: 
    max: 0.001
    min: 0.0001
    distribution: log_uniform_values
  optimizer: 
    values: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
  weight_init: 
    values: ["random", "xavier"]
  activation:
    values: ["sigmoid", "tanh", "relu"]
  momentum:
    max: 0.999
    min: 0.1
    distribution: log_uniform_values
  beta:
    max: 0.999
    min: 0.1
    distribution: log_uniform_values
  beta1:
    max: 0.999
    min: 0.1
    distribution: log_uniform_values
  beta2:
    max: 0.999
    min: 0.1
    distribution: log_uniform_values

  ```