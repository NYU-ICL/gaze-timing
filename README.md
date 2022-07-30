# Image Features Influence Reaction Time: A Learned Probabilistic Perceptual Model for Saccade Latency

[\[project page\]](https://research.nvidia.com/publication/2022-08_image-features-influence-reaction-time-learned-probabilistic-perceptual-model) [\[arXiv\]](https://arxiv.org/abs/2205.02437)

## Usage

To run the model, you need to install `numpy`, and `pytorch`. The script also imports `matplotlib` to visualize the output, but is optional.

```shell
pip install -r requirements.txt
```

The model can be run from the command line as

```shell
python model.py <CONTRAST> <FREQUENCY> <ECCENTRICITY>
```

For more options, run

```shell
python model.py -h
```

## Citation

If you use our code in your research, please cite:

Budmonde Duinkharjav, Praneeth Chakravarthula, Rachel Brown, Anjul Patney, Qi Sun
Image Features Influence Reaction Time: A Learned Probabilistic Perceptual Model for Saccade Latency.
ACM Transactions on Graphics 41(4) (Proceedings of ACM SIGGRAPH 2022)
