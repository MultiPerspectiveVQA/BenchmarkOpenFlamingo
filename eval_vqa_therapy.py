from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

import torch
import os
import copy
import csv


img_dir = '/rc_scratch/anve4082/datasets/vqa_therapy/train/'
data_dir = '/rc_scratch/anve4082/datasets/vqa_therapy/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
results_file = 'results/flamingo_4b_instruct_8shot.csv'

imgs_for_prompts = ['/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000063334.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/test/COCO_train2014_000000174211.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000420523.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000110777.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/test/COCO_train2014_000000360334.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/test/COCO_train2014_000000233956.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000172233.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/test/COCO_train2014_000000502495.jpg']

texts_for_prompts = ['<image> Question: What is on the ground behind the picther? Answers: grass, mound, dirt. Question: Do the answers of the previous question describe the same visual content? Answer: no.',
                     '<image> Question: What is the person jumping over? Answers: parking meter, meter. Question: Do the answers of the previous question describe the same visual content? Answer: yes.',
                     '<image> Question: What is on the computer screen? Answers: banana, webpage. Question: Do the answers of the previous question describe the same visual content? Answer: no.',
                     '<image> Question: What is over the elephant? Answers: blanket, umbrella. Question: Do the answers of the previous question describe the same visual content? Answer: no.',
                     '<image> Question: What is the man wearing? Answers: sweatshirt, hoodie. Question: Do the answers of the previous question describe the same visual content? Answer: yes.',
                     '<image> Question: What is the bird walking on? Answers: plants, algae. Question: Do the answers of the previous question describe the same visual content? Answer: yes.',
                     '<image> Question: Where is the man? Answers: park, under the kite. Question: Do the answers of the previous question describe the same visual content? Answer: no.',
                     '<image> Question: What is the kitten peeking out from? Answers: bowl, cup. Question: Do the answers of the previous question describe the same visual content? Answer: yes.']

def init_model():
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",clip_vision_encoder_pretrained="openai",lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",cross_attn_every_n_layers=1
        )
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    return model, image_processor, tokenizer

def init_image_prompts(image_processor):
    vision_prompt = list()
    for img in imgs_for_prompts:
        pil_img = Image.open(img)
        vision_prompt.append(image_processor(pil_img).unsqueeze(0))
    return vision_prompt

def init_text_prompt():
    lang_prompt = '<|endofchunk|>'.join(texts_for_prompts)
    return lang_prompt

def load_vqa_therapy(dir):
    vqa_therapy_train_dataset = load_dataset('imagefolder', data_dir=os.path.join(dir, 'train'), split='train')
    return vqa_therapy_train_dataset

def eval():
    model, image_processor, tokenizer = init_model()
    model.to(device)
    tokenizer.padding_side = "left"
    vision_prompt = init_image_prompts(image_processor)
    lang_prompt = init_text_prompt()
    results = list()
    train = load_vqa_therapy(data_dir)
    count = 0
    for data in tqdm(train):
        count += 1
        image, question = data['image'], data['question']
        answers = ', '.join(data['answers'])
        vision_x = vision_prompt[:]
        vision_x.append(image_processor(image).unsqueeze(0))
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        query = f'Question: {question} Answers: {answers}. Question: Do the answers of the previous question describe the same visual content? Answer:'
        lang_x = tokenizer([copy.deepcopy(lang_prompt) + query], return_tensors='pt')
        generated_text = model.generate(vision_x=vision_x.to(0),
                                        lang_x=lang_x["input_ids"].to(0),
                                        attention_mask=lang_x["attention_mask"].to(0),
                                        max_new_tokens=20,
                                        num_beams=3,
                                        )
        temp = [data['image_id'], data['question_id'], tokenizer.decode(generated_text[0])]
        results.append(temp)
    return results

def write_results(data):
    with open(results_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__ == '__main__':
    data = eval()
    write_results(data)