# Use a pipeline as a high-level helper
from transformers import pipeline
from tqdm import tqdm


def gen_image_captions(dataset):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=1)
    caption_col = list()
    for record in tqdm(dataset):
        caption = pipe(record['image'])
        caption_col.append(caption[0]['generated_text'])
    dataset = dataset.add_column('caption', caption_col)
    return dataset