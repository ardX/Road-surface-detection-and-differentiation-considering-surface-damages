from fastai.vision import *
from fastai.vision.interpret import *
from fastai.callbacks.hooks import *
from pathlib import Path
from fastai.utils.mem import *
torch.backends.cudnn.benchmark=True

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

path = Path('data/')
path.ls()
codes = np.loadtxt(path/'codes.txt', dtype=str); codes
#path_lbl = path/'labels'
path_lbl = path/'images'
path_img = path/'images'
fnames = get_image_files(path_img)

get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'

size = np.array([1080, 1920])

free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
if free > 8200: bs=8
else:           bs=4
print(f"using bs={bs}, have {free}MB of GPU RAM free")
src = (SegmentationItemList.from_folder(path_img)
       .split_none()
       .label_from_func(get_y_fn, classes=codes))
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
learn = unet_learner(data, models.resnet34)
learn.load('stage-2-weights')

results_save = 'results'
path_rst = path/results_save
path_rst.mkdir(exist_ok=True)
def save_preds(names):
    i=0
    #names = dl.dataset.items
    
    for b in names:
        img_s = fnames[i]
        img_toSave = open_image(img_s)
        img_split = f'{img_s}'
        img_split = img_split[12:]
        print(str(path_rst) +"/"+ img_split)
        predictionSave = learn.predict(img_toSave)
        predictionSave[0].save(str(path_rst) +"/"+ img_split) #Save Image
        i += 1
        print(i)
save_preds(fnames)