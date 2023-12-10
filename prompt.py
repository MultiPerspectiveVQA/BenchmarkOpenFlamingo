# package files
from gen_image_captions.image_captioner import gen_image_captions

SIMPLE_STD = '<image> Question: {question} Answer: {answer} Question: Do all the given answers for the question point to the same visual content in the image? Answer:'
SIMPLE_IMG_CAP = '<image> Context: {caption} Question: {question} Answer: {answer} Question: Do all the given answers for the question point to the same visual content in the image? Answer:'

MULTI_ANS_STD = '<image> Indicate every possible answer to the given question. Question: {question} Answer:'
MULTI_ANS_IMG_CAP = '<image> Context: {caption} Indicate every possible answer to the given question. Question: {question} Answer:'


def gen_simple_qa_prompt(record, template):
    prompt = template
    question, answer = record['question'], ', '.join(record['answers'])
    prompt = prompt.format(question=question, answer=answer)
    return {'prompt': prompt}

def gen_simple_cap_prompt(record, template):
    prompt = template
    caption, question, answer = record['caption'], record['question'], ', '.join(record['answers'])
    prompt = prompt.format(caption=caption, question=question, answer=answer)
    return {'prompt': prompt}

def gen_multi_ans_qa_prompt(record, template):
    prompt = template
    question = record['question']
    prompt = prompt.format(question=question)
    return {'prompt': prompt}

def gen_multi_ans_cap_prompt(record, template):
    prompt = template
    caption, question = record['caption'], record['question']
    prompt = prompt.format(caption=caption, question=question)
    return {'prompt': prompt}

def append_prompts(dataset, args):
    if args.test_type == 'simple' and args.prompt_type == 'std':
        return dataset.map(gen_simple_qa_prompt, batched=False, fn_kwargs={'template': SIMPLE_STD})
    elif args.test_type == 'simple' and args.prompt_type == 'img_cap':
        dataset = gen_image_captions(dataset)
        return dataset.map(gen_simple_cap_prompt, batched=False, fn_kwargs={'template': SIMPLE_IMG_CAP})
    elif args.test_type == 'multi_ans' and args.prompt_type == 'std':
        return dataset.map(gen_multi_ans_qa_prompt, batched=False, fn_kwargs={'template': MULTI_ANS_STD})
    elif args.test_type == 'multi_ans' and args.prompt_type == 'img_cap':
        dataset = gen_image_captions(dataset)
        return dataset.map(gen_multi_ans_cap_prompt, batched=False, fn_kwargs={'template': MULTI_ANS_IMG_CAP})
    else:
        raise Exception('Invalid test type/ prompt type')