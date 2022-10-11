import argparse
import datetime
import os
import time

from datetime import datetime
from os.path import isdir, exists
from PIL import Image, PngImagePlugin
from tensorflow import keras

from stable_diffusion_tf.stable_diffusion import StableDiffusion

parser = argparse.ArgumentParser()

parser.add_argument(
	"-p",
	"--prompt",
	type=str,
	nargs="?",
	default="a painting of a virus monster playing guitar",
	help="the prompt to render",
)

parser.add_argument(
	"-o",
	"--outdir",
	type=str,
	nargs="?",
	default="output",
	help="where to save the output images",
)

parser.add_argument(
	"-H",
	'--height',
	type=int,
	default=512,
	help="image height, in pixels",
)

parser.add_argument(
	"-W",
	'--width',
	type=int,
	default=512,
	help="image width, in pixels",
)

parser.add_argument(
	"-s",
	"--scale",
	type=float,
	default=7.5,
	help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

parser.add_argument(
	'-i', "--steps", type=int, default=50, help="number of ddim sampling steps"
)

parser.add_argument(
	'-S',
	"--seed",
	type=int,
	help="optionally specify a seed integer for reproducible results",
)

parser.add_argument(
	"--mp",
	default=False,
	action="store_true",
	help="Enable mixed precision (fp16 computation)",
)

parser.add_argument("-c", "--copies", type=int, default=1, help="The number of image copies to be created")
args = parser.parse_args()

if args.mp:
	print("Using mixed precision.")
	keras.mixed_precision.set_global_policy("mixed_float16")

outdir = args.outdir
if not isdir('output'):
	os.mkdir('output')
seed = args.seed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def save_image_and_prompt_to_png(image):
	global args, seed

	str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	fn = f'{str}_{seed}.png'
	path = os.path.join(outdir, fn)
	ndx = 1
	while exists(fn):
		fn = f'{str}_{seed}_{ndx}.png'
		ndx += 1
	info = PngImagePlugin.PngInfo()
	meta = f'{get_info_string()} -S{seed}'
	info.add_text('SDInfo', meta)
	info.add_text('Title', args.prompt)
	info.add_text('Seed', f'{seed}')
	image.save(path, 'PNG', pnginfo=info)
	return path

def get_info_string():
	global args

	switches = list()
	switches.append(f'{args.prompt}')
	switches.append(f'-s{args.steps}')
	switches.append(f'-W{args.width}')
	switches.append(f'-H{args.height}')
	switches.append(f'-C{args.scale}')
	# switches.append(f'-A{cfg.scheduler}')
	# if cfg.init_image:
	# 	switches.append(f'-I{cfg.init_image}')
	# if cfg.noise_strength and cfg.init_image is not None:
	# 	switches.append(f'-f{cfg.strength}')
	return ' '.join(switches)

generator = StableDiffusion(img_height=args.height, img_width=args.width, jit_compile=False)
print(f'The SEED for this batch is: {args.seed}')
tic = time.time()
seed, nd_arr = generator.generate(
	args.prompt,
	num_steps=args.steps,
	unconditional_guidance_scale=args.scale,
	temperature=1,
	batch_size=args.copies,
	seed=args.seed,
)
dur = time.time() - tic
min = int(dur / 60)
sec = dur % 60
print('>> Run stats:')
print(f'>>   {args.copies} image(s) generated in: {min}m', '%4.2fs' % sec)
print(f'>>   prompt: {args.prompt}')
print(f'>>   seed: {seed}')
# Save images
for arr in nd_arr:
	img = Image.fromarray(arr)
	fn = save_image_and_prompt_to_png(img)
	os.system(f'open {fn}')

print(f'Completed')