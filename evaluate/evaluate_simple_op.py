# Python libraries
import argparse
import pandas as pd
from tqdm import tqdm

def update_same_grd_acc(same_grd_acc, ans, prompt):
    if type(ans) != str:
        return same_grd_acc
    _, _, pred = ans.partition(prompt)
    if any(res in pred for res in ['yes', 'Yes', 'YES']):
        return same_grd_acc + 1
    else:
        return same_grd_acc
    
def update_diff_grd_acc(diff_grd_acc, ans, prompt):
    if type(ans) != str:
        return diff_grd_acc
    _, _, pred = ans.partition(prompt)
    if any(res in pred for res in ['no', 'No', 'NO']):
        return diff_grd_acc + 1
    else:
        return diff_grd_acc

def eval(args):
    data = pd.read_csv(args.results_file, header=0)
    same_grd_count, diff_grd_count, same_grd_acc, diff_grd_acc = 0, 0, 0, 0
    for _, row in tqdm(data.iterrows()):
        if row['binary_label'] == 'same':
            same_grd_count += 1
            same_grd_acc = update_same_grd_acc(same_grd_acc, row['result'], row['prompt'])
        else:
            diff_grd_count += 1
            diff_grd_acc = update_diff_grd_acc(diff_grd_acc, row['result'], row['prompt'])
    overall_accuracy = (same_grd_acc + diff_grd_acc) / (same_grd_count + diff_grd_count)
    same_grd_acc /= same_grd_count
    diff_grd_acc /= diff_grd_count
    print(f'Results')
    print(f'Overall accuracy: {overall_accuracy}')
    print(f'Same grounding accuracy: {same_grd_acc}')
    print(f'Diff grounding accuracy: {diff_grd_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='evalue simple outputs',
        description='Module to help evaluate simple format results for MultiPurpose VQA'
        )
    parser.add_argument('--results_file', required=True, type=str, help='Absolute path of the results file')
    args = parser.parse_args()
    eval(args)