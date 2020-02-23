import pickle
import numpy as np
import sys
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import torchvision.models as models
if sys.platform == "darwin": # Apple
   import matplotlib
   matplotlib.use("TkAgg")
   from matplotlib import pyplot as plt
elif 'linux' in sys.platform:
   import matplotlib
   matplotlib.use("agg")
   from matplotlib import pyplot as plt
else:
    from matplotlib import pyplot as plt

np.random.seed(0)

def all_bytes_to_strings(d):
    """
    Takes a item (list, dictionary, or byte string), and returns
    a new item where all the byte strings have been recursively
    transformed into regular strings
    """
    if type(d) == dict:
        new_dict = {}
        for k, v in d.items():
            k = all_bytes_to_strings(k)
            v = all_bytes_to_strings(v)
            new_dict[k] = v
        return new_dict
    if type(d) == list:
        ls = []
        for item in d:
            ls.append(
                all_bytes_to_strings(item)
            )
        return ls
    if type(d) == bytes:
        return str(d, 'utf-8')
    else:
        return d

def unpickle_cifar(file):
    """
    Loads the downloaded CIFAR-100 data from file
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    dict = all_bytes_to_strings(dict)
    return dict

def filter_by_label(data, labels, filter_label):
    """
    Returns the subset of the data with the desired label
    """
    n = len(data)
    indices = [idx for idx in range(n) if labels[idx] == filter_label]
    return data[indices]

def visualize_image(img):
    """ Displays the image """
    img = img.transpose(1,2,0)
    plt.imshow(img)
    plt.show()

def extract_img_features(img_data):
    """ Extracts features from image data with pretrained AlexNet """
    n = len(img_data)
    img_data = img_data / np.max(img_data) # normalize to [0,1] range
    img_data = img_data.reshape(n, 3, 32, 32)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            inplace=True
        )
    ])
    net = models.alexnet(pretrained=True)
    X = torch.FloatTensor(img_data)
    transformed_X = torch.FloatTensor(np.zeros((n, 3, 64, 64)))
    y = torch.zeros(len(img_data))
    for i in range(len(X)):
        transformed_X[i] = transform(X[i])
    X = transformed_X
    # feed through network
    net.eval()
    dataset = TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100)
    # build output dataset
    X = []
    for (inputs, labels) in dataloader:
        with torch.no_grad():
            outputs = net(inputs)
        for pred in outputs:
            X.append(pred.numpy())
    X = np.array(X)
    return X


def save_cifar_data(filename='data/cifar.pkl'):
    """ Saves the processed cifar dataset """
    d = unpickle_cifar('data/cifar-100-python/train')
    meta = unpickle_cifar('data/cifar-100-python/meta')
    coarse_label_names = meta['coarse_label_names']
    coarse_label_map = dict([(coarse_label_names[idx], idx) for idx in range(len(coarse_label_names))])
    fine_label_names = meta['fine_label_names']
    fine_label_map = dict([(fine_label_names[idx], idx) for idx in range(len(fine_label_names))])
    all_data = d['data']
    all_coarse_labels = d['coarse_labels']
    all_fine_labels = d['fine_labels']
    fish = filter_by_label(all_data, all_coarse_labels, coarse_label_map['fish'])
    aq_mammals = filter_by_label(all_data, all_coarse_labels, coarse_label_map['aquatic_mammals'])
    sub_fish = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
    sub_aq_mammals = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
    remove_sizes = [100, 200, 300, 400, 500]
    data = []
    labels = []
    sublabels = []
    for i in range(len(remove_sizes)):
        remove_size = remove_sizes[i]
        fish_data = filter_by_label(all_data, all_fine_labels, fine_label_map[sub_fish[i]])
        mammal_data = filter_by_label(all_data, all_fine_labels, fine_label_map[sub_aq_mammals[i]])
        selected_fish_indices = np.random.choice(range(len(fish_data)), size=remove_size, replace=False)
        selected_mammal_indices = np.random.choice(range(len(mammal_data)), size=remove_size, replace=False)
        data.append(fish_data[selected_fish_indices])
        data.append(mammal_data[selected_mammal_indices])
        labels.append(np.zeros(remove_size) - 1) # DO WE WANT y in {-1, 1} or {0, 1}?
        labels.append(1 + np.zeros(remove_size))
        sublabels.append(fine_label_map[sub_fish[i]] + np.zeros(remove_size))
        sublabels.append(fine_label_map[sub_aq_mammals[i]] + np.zeros(remove_size))
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    sublabels = np.concatenate(sublabels, axis=0)
    data = {
        'X': extract_img_features(data),
        'y': labels,
        'sublabels': sublabels,
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print('Data saved to {}.'.format(filename))

def load_all_cifar_data():
    d = unpickle_cifar('data/cifar-100-python/train')
    meta = unpickle_cifar('data/cifar-100-python/meta')
    coarse_label_names = meta['coarse_label_names']
    coarse_label_map = dict([(coarse_label_names[idx], idx) for idx in range(len(coarse_label_names))])
    all_data = d['data']
    all_coarse_labels = d['coarse_labels']
    fish = filter_by_label(all_data, all_coarse_labels, coarse_label_map['fish'])
    aq_mammals = filter_by_label(all_data, all_coarse_labels, coarse_label_map['aquatic_mammals'])
    data = []
    labels = []
    for row in fish:
        data.append(row)
        labels.append(-1) # DO WE WANT y in {-1, 1} or {0, 1}?
    for row in aq_mammals:
        data.append(row)
        labels.append(1)
    data = np.array(data)
    labels = np.array(labels)
    X_data = extract_img_features(data)
    # n = len(X_data)
    # X_data = np.append(X_data, 1+np.zeros((n,1)), axis=1)
    data = {
        'X': X_data,
        'y': labels,
    }
    return data

def alt_load_all_cifar_data():
    d = unpickle_cifar('data/cifar-100-python/train')
    meta = unpickle_cifar('data/cifar-100-python/meta')
    coarse_label_names = meta['coarse_label_names']
    coarse_label_map = dict([(coarse_label_names[idx], idx) for idx in range(len(coarse_label_names))])
    all_data = d['data']
    all_coarse_labels = d['coarse_labels']
    fish = filter_by_label(all_data, all_coarse_labels, coarse_label_map['fish'])
    aq_mammals = filter_by_label(all_data, all_coarse_labels, coarse_label_map['aquatic_mammals'])
    data = []
    labels = []
    for row in fish:
        data.append(row)
        labels.append(-1) # {-1, 1} labels
    for row in aq_mammals:
        data.append(row)
        labels.append(1)
    data = np.array(data)
    labels = np.array(labels)
    X_data = extract_img_features(data)
    n = len(X_data)
    pos_idx = [idx for idx in range(n) if labels[idx] == 1]
    outliers = np.random.choice(pos_idx, 100, replace=False)
    X_data[outliers] = X_data[outliers]*25 # ARTIFICAL INTRODUCTION OF OUTLIERS
    data = {
        'X': X_data,
        'y': labels,
        'outliers': outliers,
    }
    return data

def load_cifar_data(filename='data/cifar.pkl'):
    """ Loads the saved data from a .pkl file """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
