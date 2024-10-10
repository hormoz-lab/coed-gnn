import argparse
import logging

parser = argparse.ArgumentParser()

### Dataset Args
parser.add_argument("--dataset", type=str, help="Name of dataset", default="chameleon")
parser.add_argument("--dataset-directory", type=str, help="Directory to save datasets", default="../datasets")

### Preprocessing Args
parser.add_argument("--undirected", action="store_true", help="Whether to use undirected version of graph")
parser.add_argument("--remove-existing-self-loop", action="store_true", help="Whether to remove existing self loops")

### Model Args
parser.add_argument("--model", type=str, help="Model type", default="CoED")
parser.add_argument("--hidden-dimension", type=int, help="Hidden dimension of model", default=64)
parser.add_argument("--num-layers", type=int, help="Number of GNN layers", default=2)
parser.add_argument("--dropout-rate", type=float, help="Feature dropout", default=0.0)
parser.add_argument("--alpha", type=float, help="Direction convex combination params", default=0.5)
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--jumping-knowledge", type=str, choices=["max", "cat", "None"], default="None")
parser.add_argument("--self-feature-transform", action="store_true", help="Whether to transform self feature")
parser.add_argument("--self-loop", action="store_true", help="Whether to mix self feature to directional messages")
parser.add_argument("--layer-wise-theta", action="store_true", help="Whether use independent edge directions at each layer")

### Training Args
parser.add_argument("--learning-rate", type=float, help="Learning rate", default=0.001)
parser.add_argument("--weight-decay", type=float, help="Weight decay", default=0.0)
parser.add_argument("--theta-learning-rate", type=float, help="Theta learning rate", default=0.01)
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=100)
parser.add_argument("--use-encoding", action="store_true", help="Whether to use positional/structural encoding")
parser.add_argument("--pe_type", type=str, help="eigenvector/electrostatic/None", default="None")
parser.add_argument("--pe_dim", type=int, help="Dimension of encoding", default=10)
parser.add_argument("--store-theta", action="store_true", help="Whether to store phase angles as they get optimized")
parser.add_argument("--print-interval",  type=int, help="Interval at which training results are printed to console", default=50)


### System Args
parser.add_argument("--use-best-hyperparams", action="store_true", help="Whether to use previously found best hyperparameters")
parser.add_argument("--gpu-idx", type=int, help="Indexes of gpu to run program on", default=0)

args = parser.parse_args()
for key, value in vars(args).items(): 
    if value == "None":
        setattr(args, key, None) 