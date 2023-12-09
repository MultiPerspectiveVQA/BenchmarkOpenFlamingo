# Python libraries
import copy
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from tqdm import tqdm
from PIL import Image

# package files
from few_shots import IMGS_4SHOTS, IMGS_8SHOTS, SIMPLE_STD_4SHOTS, SIMPLE_CAP_4SHOTS, MULTI_ANS_STD_4SHOTS, MULTI_ANS_CAP_4SHOTS, SIMPLE_STD_8SHOTS, SIMPLE_CAP_8SHOTS, MULTI_ANS_STD_8SHOTS, MULTI_ANS_CAP_8SHOTS


def example_mapper(args):
    if args.shots == 4 and args.test_type == 'simple' and args.prompt_type == 'std':
        return IMGS_4SHOTS, SIMPLE_STD_4SHOTS
    elif args.shots == 4 and args.test_type == 'simple' and args.prompt_type == 'cap':
        return IMGS_4SHOTS, SIMPLE_CAP_4SHOTS
    elif args.shots == 4 and args.test_type == 'multi_ans' and args.prompt_type == 'std':
        return IMGS_4SHOTS , MULTI_ANS_STD_4SHOTS
    elif args.shots == 4 and args.test_type == 'multi_ans' and args.prompt_type == 'cap':
        return IMGS_4SHOTS , MULTI_ANS_CAP_4SHOTS
    elif args.shots == 8 and args.test_type == 'simple' and args.prompt_type == 'std':
        return IMGS_8SHOTS, SIMPLE_STD_8SHOTS
    elif args.shots == 8 and args.test_type == 'simple' and args.prompt_type == 'cap':
        return IMGS_8SHOTS, SIMPLE_CAP_8SHOTS
    elif args.shots == 8 and args.test_type == 'multi_ans' and args.prompt_type == 'std':
        return IMGS_8SHOTS , MULTI_ANS_STD_8SHOTS
    elif args.shots == 8 and args.test_type == 'multi_ans' and args.prompt_type == 'cap':
        return IMGS_8SHOTS , MULTI_ANS_CAP_8SHOTS

def load_image_examples(imgs, image_processor):
    image_examples = list()
    for img in imgs:
        pil_img = Image.open(img)
        image_examples.append(image_processor(pil_img).unsqueeze(0))
    return image_examples

def load_lang_examples(texts):
    lang_examples = '<|endofchunk|>'.join(texts)
    return lang_examples

def get_model():
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",clip_vision_encoder_pretrained="openai",lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",cross_attn_every_n_layers=1
        )
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    return model, image_processor, tokenizer

def get_results(dataset, args):
    model, image_processor, tokenizer = get_model()
    imgs, texts = example_mapper(args)
    vision_examples = load_image_examples(imgs, image_processor)
    text_examples = load_lang_examples(texts)
    results = []
    for record in tqdm(dataset):
        output = [record['image_id'], record['question_id'], record['image_filename'], record['binary_label'], record['question'], '| '.join(record['answers']), record['prompt']]
        image, prompt = record['image'], record['prompt']
        vision_x = vision_examples[:]
        vision_x.append(image_processor(image).unsqueeze(0))
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        lang_x = tokenizer([copy.deepcopy(text_examples) + prompt], return_tensors='pt')
        generated_text = model.generate(vision_x=vision_x.to(0),
                                        lang_x=lang_x["input_ids"].to(0),
                                        attention_mask=lang_x["attention_mask"].to(0),
                                        max_new_tokens=20,
                                        num_beams=3,
                                        )
        output.append(tokenizer.decode(generated_text[0]))
        results.append(output)
    return results

