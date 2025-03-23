import argparse

from experiments.config import hyperparam_options

def main():
    # Obtaining hyperparameters based off of 'config_no'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_no', type=int, required=True)
    args = parser.parse_args()
    params = hyperparam_options[args.config_no]
    print(params)

if __name__ == '__main__':
    main()