#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import torch_rbf as rbf


def build_model(path = None):
    LAYER_WIDTHS = [3, 1]
    LAYER_CENTRES = [20]
    model = GazeTimingModel(LAYER_WIDTHS, LAYER_CENTRES, rbf.gaussian)
    if path is not None:
        model.load(path)
    return model


def eval_rate(c, f, e, model_path = "./io/model.pth"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise Exception("Model file not found at %s", model_path)
    model = build_model(model_path)
    model.eval()
    inp = np.stack([c, f, e], axis=-1)
    rate = model(torch.tensor(inp, dtype=torch.float32)).detach().numpy()
    return rate[:, 0]


def invgauss_pdf(t, alpha, rate, epsilon=1e-12):
    return alpha / (2 * np.pi * t ** 3 + epsilon) ** 0.5 * np.exp(-(alpha - rate * t) ** 2 / (2 * t + epsilon))


class GazeTimingModel(nn.Module):
    def __init__(self, layer_widths, layer_centres, basis_func):
        super(GazeTimingModel, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(rbf.RBF(layer_widths[i], layer_centres[i], basis_func))
            self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i+1]))

    def forward(self, x):
        x /= torch.tensor([[1.0, 4.0, 20.0]], dtype=torch.float32)
        out = x
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return out

    def load(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("C", metavar="CONTRAST", type=float,
                        help="michelson contrast of stimulus; valid between [0, 1]")
    parser.add_argument("F", metavar="FREQUENCY", type=float,
                        help="frequency of stimulus in cpd; valid between [0, 4]")
    parser.add_argument("E", metavar="ECCENTRICITY", type=float,
                        help="retinal eccentricity of stimulus in degrees; valid between [0, 20]")
    parser.add_argument("--model_path", type=str, default="./io/model.pth",
                        help="path to trained model")
    parser.add_argument("--conf_threshold", type=float, default="3.7",
                        help="confidence threshold alpha")
    parser.add_argument("--baseline_rt", type=float, default=300,
                        help="baseline reaction time for c=1, f=1, e=0 condition in ms")
    parser.add_argument("--plot", action="store_true",
                        help="visualize the pdf function (requires matplotlib)")
    opt = parser.parse_args()

    rate = eval_rate([opt.C], [opt.F], [opt.E], model_path = opt.model_path)
    print("Mean RT: ", 1/rate[0] * opt.baseline_rt)
    if opt.plot:
        norm_t = np.linspace(0, 2.5, 50)
        t = norm_t * opt.baseline_rt
        pdf = invgauss_pdf(norm_t, opt.conf_threshold, opt.conf_threshold * rate)
        import matplotlib.pyplot as plt
        plt.plot(t, pdf)
        plt.title("Predicted PDF")
        plt.xlabel("Reaction Time (ms)")
        plt.ylabel("Probability Density")
        plt.show()
