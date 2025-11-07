
'''
import numpy as np, matplotlib.pyplot as plt, json
imgs = np.load('car_dd_out/images.npy', mmap_mode='r')
masks = np.load('car_dd_out/instance_masks.npz', allow_pickle=True)['masks']
with open('car_dd_out/targets.json') as f: meta = json.load(f)['targets']
i = 1
plt.imshow(imgs[i]); plt.show()
plt.imshow(masks[i].sum(axis=0), alpha=0.4); plt.show()
print(meta[list(meta.keys())[i]])
'''
import numpy as np, matplotlib.pyplot as plt, json

imgs = np.load('car_dd_out/images.npy', mmap_mode='r')
mask_bundle = np.load('car_dd_out/instance_masks.npz', allow_pickle=True)['masks']
with open('car_dd_out/targets.json') as f:
    targets = json.load(f)['targets']

i = 5
img = imgs[i]
mask = mask_bundle[i].sum(axis=0)

fig, ax = plt.subplots()
ax.imshow(img)
ax.imshow(mask, alpha=0.4, cmap='autumn')
ax.axis('off')
print(targets[list(targets.keys())[i]])
plt.show()