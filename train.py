import json
import argparse
import importlib
from solver import Solver
import torchvision
# print(torchvision.__version__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='HDAN')
    parser.add_argument("--ckpt_name", type=str, default='test_X4')
    parser.add_argument("--print_interval", type=int, default=5000)
    parser.add_argument("--train_data_path", type=str, default="./DF2K_x4_fixed.h5")
    parser.add_argument("--valid_data_path", type=str, default="./dataset/DIV2K_valid")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--sample_dir", type=str, default="sample")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--pre_trained", type=bool, default=False)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--shave", type=int, default=4)
    parser.add_argument("--scale", type=int, default=4)
    # parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--verbose", action="store_true", default="store_true")
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=500000)
    # parser.add_argument("--decay", type=int, default=30000)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--clip", type=float, default=10.0)
    # parser.add_argument("--valid_max_size", type=int, default=256")
    parser.add_argument("--loss_fn", type=str, choices=["MSE", "L1", "SmoothL1","Charbonnier", 'L1_Charb'], default="L1")
    # parser.add_argument("--optimizer", type=str, choices=["AdamP", "Adam"], default="AdamP")
    # parser.add_argument("--T_max", type=int, default=200000)
    # parser.add_argument("--eta_min", type=float, default=1e-6)
    return parser.parse_args()

def main(cfg):

    net = importlib.import_module("{}".format(cfg.model)).Net
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    solver = Solver(net, cfg)
    solver.fit()

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
