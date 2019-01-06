import csv
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

class ChalDataset(Dataset):
    def __init__(self, args):
        self.args = args

        print("Dataloader loading csv file: {}".format(args.input_csv))

        csvfile = csv.reader(open(args.input_csv,'r'))
        header = next(csvfile)
        
        data = []
        formulaA = []
        formulaB = []
        stabilityVec = []
        for row in csvfile:
            formulaA.append(row[0])
            formulaB.append(row[1])
            data.append(np.array([np.float(x) for x in row[2:-1]]))
            stabilityVec.append(np.array([np.float(x) for x in row[-1][1:-1].split(',')]))
        
        stabilityVec = np.array(stabilityVec)
        
        formulas = formulaA + formulaB
        formulas = list(set(formulas))
        
        # -- /!\ need to save the dict as the ordering may difer at each run
        formula2int = {}
        int2formula = {}
        for i, f in enumerate(formulas):
            formula2int[f] = i
            int2formula[i] = f
        
        formulaAint = np.array([formula2int[x] for x in formulaA])
        formulaBint = np.array([formula2int[x] for x in formulaB])
        data = np.array(data)
        data = np.concatenate((formulaAint[:,None], formulaBint[:,None], data), axis=1)
        
        if args.normalize:
            data = normalize(data, axis=1)

        y_true = stabilityVec[:,1:-1]
        X_train, X_test, y_train, y_test = train_test_split(data, y_true,
                                                            test_size=args.test_size,
                                                            shuffle=True,
                                                            random_state=42)
        self.data = data
        self.stabilityVec = stabilityVec

        self.X_train = torch.from_numpy(X_train).float()
        self.y_train = torch.from_numpy(y_train).float()
        self.X_test = torch.from_numpy(X_test).float()
        self.y_test = torch.from_numpy(y_test).float()

        self.num_data_points = {}
        self.num_data_points['train'] = len(X_train)
        self.num_data_points['test'] = len(X_test)
        
        self._split = 'train'

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        self._split = split

    # ------------------------------------------------------------------------
    # methods to override - __len__ and __getitem__ methods
    # ------------------------------------------------------------------------

    def __len__(self):
        return self.num_data_points[self._split]

    def __getitem__(self, idx):
        dtype = self._split
        item = {'index': idx}
        item['features'] = self.X_train[idx,:]
        item['outputs'] = self.y_train[idx,:]
        return item

    #-------------------------------------------------------------------------
    # collate function utilized by dataloader for batching
    #-------------------------------------------------------------------------

    def collate_fn(self, batch):
        dtype = self._split
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        out = {}
        for key in merged_batch:
            if key in {'index'}:
                out[key] = merged_batch[key]
            else:
                out[key] = torch.stack(merged_batch[key], 0)

        batch_keys = ['features', 'outputs']
        return {key: out[key] for key in batch_keys}


