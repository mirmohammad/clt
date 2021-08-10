import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="graph generator")
parser.add_argument('-tl', required=True, type=str, help='path to training losses')
parser.add_argument('-vl', required=True, type=str, help='path to validation losses')
parser.add_argument('-ta', required=True, type=str, help='path to training accuracies')
parser.add_argument('-va', required=True, type=str, help='path to validation accuracies')

args = parser.parse_args()

train_loss = pd.read_csv(args.tl, delimiter=',')
valid_loss = pd.read_csv(args.vl, delimiter=',')
train_acc = pd.read_csv(args.ta, delimiter=',')
valid_acc = pd.read_csv(args.va, delimiter=',')

print(train_loss)
