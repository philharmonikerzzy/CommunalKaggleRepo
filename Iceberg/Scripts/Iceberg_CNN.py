import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pandas.read_json("train.json")

# data.columns
# data.columns.values
# for items in data.columns.values:
    # print(data[items].head())
	

data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')

band_1 = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
full_img = np.stack([band_1, band_2], axis=1)

# full_img.shape

# plt.imshow(full_img[2,0,:,:])
# plt.axis('off')
# print("Image Label: ", 1)

# full_img[0,1]


def savetxt(filename, ndarray):
    if not os.path.isfile(filename):
        print("Saving", filename )
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:
        print("File already exists", filename)