from tqdm import tqdm
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
from classify import load, classify

search = "eye"#@param {type:"string"}
things=[search]


load(things)
blocks = []
scores = []
rescale = 512#@param
chunk_size=128#@param

image = Image.open("hello.jpg")
w,h = image.size
image = image.resize((rescale,rescale))
npimg = np.array(image)
big_chunks=False#@param {type:"boolean"}

def block(x,y):
    b = []
    for i in range(chunk_size-1):
        b.append(npimg[x+i][y:y+chunk_size])
    b = np.array(b)
    b = Image.fromarray(b)
    b.save("image.png")
    return b, classify("image.png", return_raw=True)[0]
blocks = []
scores = []
ii = []
jj = []
# top row

if(big_chunks):
    iterate = int(size/chunk_size-1)
else:
    iterate = rescale

iterate = 20
for i in tqdm(range(iterate)):
    for j in range(iterate):
        if(big_chunks):
            b,c = block(i*chunk_size,j*chunk_size)
            ii.append(i*chunk_size)
            jj.append(j*chunk_size)
        else:
            b,c = block(i,j)
            ii.append(i)
            jj.append(j)
        blocks.append(b)
        scores.append(c)
best_index = scores.index(max(scores)) 
iii = ii[best_index]
jjj = jj[best_index]
score = scores[best_index]
print("top left x: {} | top left y {} | similarity: {}".format(iii,jjj,score))
blocks[scores.index(max(scores))].resize((int(w/8)*4,int(h/8)*4))