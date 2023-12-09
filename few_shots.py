IMGS_4SHOTS = ['/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000063334.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000425465.jpg','/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000556637.jpg','/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000172233.jpg']

IMGS_8SHOTS = ['/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000063334.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000425465.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000420523.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000110777.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000556637.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000048944.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000172233.jpg', '/rc_scratch/anve4082/datasets/vqa_therapy/train/COCO_train2014_000000456648.jpg']

SIMPLE_STD_4SHOTS = ['<image> Question: What is on the ground behind the picther? Answer: grass, mound, dirt. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.',
                     '<image> Question: What is on the rail? Answer: train car, boxcar. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.',
                     '<image> Question: What is the man eating? Answer: doughnut, donut. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.',
                     '<image> Question: Where is the man? Answer: park, under the kite. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.']

SIMPLE_CAP_4SHOTS = ['<image> Caption: pitcher throwing a baseball from mound during a baseball game. Question: What is on the ground behind the picther? Answer: grass, mound, dirt. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.',
                     '<image> Caption: there is a rusted train car sitting on the tracks. Question: What is on the rail? Answer: train car, boxcar. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.',
                     '<image> Caption: man eating a doughnut outside in front of a brick wall. Question: What is the man eating? Answer: doughnut, donut. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.',
                     '<image> Caption: there is a man holding a kite in the air near a bench. Question: Where is the man? Answer: park, under the kite. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.']

MULTI_ANS_STD_4SHOTS = ['<image> Indicate every possible answer to the given question. Question: What is on the ground behind the picther? Answer: grass, mound, dirt.',
                     '<image> Indicate every possible answer to the given question. Question: What is on the rail? Answer: train car, boxcar.',
                     '<image> Context: man eating a doughnut outside in front of a brick wall. Indicate every possible answer to the given question. Question: What is the man eating? Answer: doughnut, donut.',
                     '<image> Indicate every possible answer to the given question. Question: Where is the man? Answer: park, under the kite.']

MULTI_ANS_CAP_4SHOTS = ['<image> Context: pitcher throwing a baseball from mound during a baseball game. Indicate every possible answer to the given question. Question: What is on the ground behind the picther? Answer: grass, mound, dirt.',
                     '<image> Context: there is a rusted train car sitting on the tracks. Indicate every possible answer to the given question. Question: What is on the rail? Answer: train car, boxcar.',
                     '<image> Context: man eating a doughnut outside in front of a brick wall. Indicate every possible answer to the given question. Question: What is the man eating? Answer: doughnut, donut.',
                     '<image> Context: there is a man holding a kite in the air near a bench. Indicate every possible answer to the given question. Question: Where is the man? Answer: park, under the kite.']


SIMPLE_STD_8SHOTS = ['<image> Question: What is on the ground behind the picther? Answer: grass, mound, dirt. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.',
                     '<image> Question: What is on the rail? Answer: train car, boxcar. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.',
                     '<image> Question: What is on the computer screen? Answer: banana, webpage. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.',
                     '<image> Question: What is over the elephant? Answer: blanket, umbrella. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.',
                     '<image> Question: What is the man eating? Answer: doughnut, donut. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.',
                     '<image> Question: What is the man holding? Answer: tennis racket, racket. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.',
                     '<image> Question: Where is the man? Answer: park, under the kite. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.',
                     '<image> Question: What is this person riding? Answer: motorcycle, moped. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.']

SIMPLE_CAP_8SHOTS = ['<image> Caption: pitcher throwing a baseball from mound during a baseball game. Question: What is on the ground behind the picther? Answer: grass, mound, dirt. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.',
                     '<image> Caption: there is a rusted train car sitting on the tracks. Question: What is on the rail? Answer: train car, boxcar. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.',
                     '<image> Caption: there is a banana and a strawberry on a laptop computer. Question: What is on the computer screen? Answer: banana, webpage. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.',
                     '<image> Caption: elephants are carrying people on their backs and carrying umbrellas. Question: What is over the elephant? Answer: blanket, umbrella. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.',
                     '<image> Caption: man eating a doughnut outside in front of a brick wall. Question: What is the man eating? Answer: doughnut, donut. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.',
                     '<image> Caption: tennis player with a red shirt and a black hat. Question: What is the man holding? Answer: tennis racket, racket. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.',
                     '<image> Caption: there is a man holding a kite in the air near a bench. Question: Where is the man? Answer: park, under the kite. Question: Do all the given answers for the question point to the same visual content in the image? Answer: no.',
                     '<image> Caption: there is a man riding a motorcycle down the street with a bag on the back. Question: What is this person riding? Answer: motorcycle, moped. Question: Do all the given answers for the question point to the same visual content in the image? Answer: yes.']

MULTI_ANS_STD_8SHOTS = ['<image> Indicate every possible answer to the given question. Question: What is on the ground behind the picther? Answer: grass, mound, dirt.',
                     '<image> Indicate every possible answer to the given question. Question: What is on the rail? Answer: train car, boxcar.',
                     '<image> Indicate every possible answer to the given question. Question: What is on the computer screen? Answer: banana, webpage.',
                     '<image> Indicate every possible answer to the given question. Question: What is over the elephant? Answer: blanket, umbrella.',
                     '<image> Indicate every possible answer to the given question. Question: What is the man eating? Answer: doughnut, donut.',
                     '<image> Indicate every possible answer to the given question. Question: What is the man holding? Answer: tennis racket, racket.',
                     '<image> Indicate every possible answer to the given question. Question: Where is the man? Answer: park, under the kite.',
                     '<image> Indicate every possible answer to the given question. Question: What is this person riding? Answer: motorcycle, moped.']

MULTI_ANS_CAP_8SHOTS = ['<image> Context: pitcher throwing a baseball from mound during a baseball game. Indicate every possible answer to the given question. Question: What is on the ground behind the picther? Answer: grass, mound, dirt.',
                     '<image> Context: there is a rusted train car sitting on the tracks. Indicate every possible answer to the given question. Question: What is on the rail? Answer: train car, boxcar.',
                     '<image> Context: there is a banana and a strawberry on a laptop computer. Indicate every possible answer to the given question. Question: What is on the computer screen? Answer: banana, webpage.',
                     '<image> Context: elephants are carrying people on their backs and carrying umbrellas. Indicate every possible answer to the given question. Question: What is over the elephant? Answer: blanket, umbrella.',
                     '<image> Context: man eating a doughnut outside in front of a brick wall. Indicate every possible answer to the given question. Question: What is the man eating? Answer: doughnut, donut.',
                     '<image> Context: tennis player with a red shirt and a black hat. Indicate every possible answer to the given question. Question: What is the man holding? Answer: tennis racket, racket.',
                     '<image> Context: there is a man holding a kite in the air near a bench. Indicate every possible answer to the given question. Question: Where is the man? Answer: park, under the kite.',
                     '<image> Context: there is a man riding a motorcycle down the street with a bag on the back. Indicate every possible answer to the given question. Question: What is this person riding? Answer: motorcycle, moped.']