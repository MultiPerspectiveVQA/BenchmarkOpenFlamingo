# Python libraries
import argparse
import os
import pandas as pd

# package files
import constants
from load_dataset import load_vqa_therapy
from prompt import append_prompts
from infer import get_results

def write_results(outputs, filename):
    df = pd.DataFrame(outputs, columns=['image_id','question_id','image_filename','binary_label','question','answers','prompt','result'])
    df.to_csv(os.path.join(constants.OUTPUTS, filename))

def main(args):
    print('loading dataset')
    dataset = load_vqa_therapy(args.split)
    print('Building prompts')
    dataset = append_prompts(dataset, args)
    print('Fecthing results')
    results = get_results(dataset, args)
    print('Writing results')
    write_results(results, args.output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Module to help prompt open flamingo for  multi-perspective vqa'
        )
    parser.add_argument('--split', choices=['train','val'], required=True, help='vqa_therapy dataset split. train/val')
    parser.add_argument('--test_type', choices=['simple', 'multi_ans'], required=True, help='Choose between yes/no style or generating all possible answers')
    parser.add_argument('--prompt_type', choices=['std', 'img_cap', 'cot'], required=True, help='Choose prompt type between standard, image caption as context, and chain of thought')
    parser.add_argument('--shots', choices=[4, 8], required=True, type=int, help='Choose 4shot examples or 8shot examples')
    parser.add_argument('--output_filename', required=True, type=str, help='Filename to store the results. Results will be stored in outputs dir')
    args = parser.parse_args()
    main(args)