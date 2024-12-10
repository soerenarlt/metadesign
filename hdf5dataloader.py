import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import numpy as np
class HDF5Dataset(Dataset):
    def __init__(self, file_path, src_key='state', tgt_key='code', max_src_len=640, max_tgt_len=640, pad_token=0, pos_only=False, minus_token=41, exclude_trivial=False):
        self.file_path = file_path
        self.src_key = src_key
        self.tgt_key = tgt_key
        
        with h5py.File(self.file_path, 'r') as file:
            self.src_data = file[src_key]
            self.tgt_data = file[tgt_key]
            self.src_data = np.array(self.src_data)
            self.tgt_data = np.array(self.tgt_data)
            # self.num_kets = file['num_kets']
            # self.num_kets = np.array(self.num_kets)

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_token = pad_token

        # print(self.src_data.shape)
        # print(self.tgt_data.shape)
        tt = time.time()
        #filter out sequences that are too long (non padding tokens)
        src_lens = np.array([len(np.where(self.src_data[i] != self.pad_token)[0]) for i in range(len(self.src_data))])
        tgt_lens = np.array([len(np.where(self.tgt_data[i] != self.pad_token)[0]) for i in range(len(self.tgt_data))])
        cond = (src_lens <= self.max_src_len) & (tgt_lens <= self.max_tgt_len)
        # if pos_only:
        #     #check if the target sequence contains any minus tokens
        #     cond = cond & (np.sum(self.tgt_data == minus_token, axis=1) == 0)
        # if exclude_trivial:
        #     #remove last column
        #     self.num_kets = self.num_kets[:,:-1]
        #     #sum of num_kets is 3
        #     cond = cond & (np.sum(self.num_kets, axis=1) > 3)

        self.src_data = self.src_data[cond][:,:self.max_src_len]
        self.tgt_data = self.tgt_data[cond][:,:self.max_tgt_len]
        # print(f'Time taken to filter out long sequences: {time.time() - tt}',flush=True)
        # print(self.src_data.shape)
        # print(self.tgt_data.shape)


        
    def __getitem__(self, index):
        src = torch.tensor(self.src_data[index]).to(torch.int64)
        tgt = torch.tensor(self.tgt_data[index]).to(torch.int64)
        return src, tgt
    
    def __len__(self):
        return len(self.src_data)
    


class HDF5DataLoader():
    def __init__(self, dataset, batch_size, local_rank, world_size, device, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.local_rank = local_rank
        self.world_size = world_size
        self.n_samples = len(dataset)
        self.n_batches = self.n_samples // (self.batch_size*self.world_size)
        self.n_samples = self.n_batches * self.batch_size * self.world_size
        #print(f'Number of samples: {self.n_samples}')
        self.epoch = 0
        self.batch_idx = 0
        self.device = device
        self.device_type = 'cuda' if 'cuda' in device else 'cpu'
        if self.shuffle: self._shuffle()

    def _shuffle(self):
        random_inds = np.random.permutation(self.n_samples)
        self.dataset.src_data = self.dataset.src_data[random_inds]
        self.dataset.tgt_data = self.dataset.tgt_data[random_inds]
        
    def get_next(self):
        ix = np.arange(self.batch_size) #[0,1,2,3]
        ix = ix*self.world_size + self.local_rank #[0,8,16,24]
        ix = ix + self.batch_idx*self.batch_size*self.world_size #[0,8,16,24] + 0*4*8 = [0,8,16,24]
        # print(f'ix: {ix}',flush=True)
        src = self.dataset[ix][0]
        x = self.dataset[ix][1][:,:-1]
        y = self.dataset[ix][1][:,1:]
        # print(f'src: {src.shape}, x: {x.shape}, y: {y.shape}')
        #check if tensor or numpy array
        # print(f'src type: {type(src)}')
        # print(f'x type: {type(x)}')
        # print(f'y type: {y}')


        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            src, x, y = src.pin_memory().to(self.device, non_blocking=True), x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            src, x, y = src.to(self.device), x.to(self.device), y.to(self.device)

        self.batch_idx += 1
        if self.batch_idx == self.n_batches:
            self.batch_idx = 0
            self.epoch += 1
            if self.shuffle: self._shuffle()

        return src, x, y
    
    def get_batch(self):
        random_batch_idx = np.random.randint(0, self.n_batches)
        ix = np.arange(self.batch_size) #[0,1,2,3]
        ix = ix*self.world_size + self.local_rank
        ix = ix + random_batch_idx*self.batch_size*self.world_size

        src = self.dataset[ix][0]
        x = self.dataset[ix][1][:,:-1]
        y = self.dataset[ix][1][:,1:]
        # print(f'src: {src.shape}, x: {x.shape}, y: {y.shape}')

        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            src, x, y = src.pin_memory().to(self.device, non_blocking=True), x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            src, x, y = src.to(self.device), x.to(self.device), y.to(self.device)

        return src, x, y
        

    def get_one(self, i=None):
        if i is None:
            i = np.random.randint(0, self.n_samples)
        else:
            assert i < self.n_samples
        src = self.dataset[i][0].unsqueeze(0)
        x = self.dataset[i][1][:-1].unsqueeze(0)
        y = self.dataset[i][1][1:].unsqueeze(0)
        # print(f'src: {src.shape}, x: {x.shape}, y: {y.shape}')

        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            src, x, y = src.pin_memory().to(self.device, non_blocking=True), x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            src, x, y = src.to(self.device), x.to(self.device), y.to(self.device)

        return src, x, y  
    


if __name__ == '__main__':
    BATCHSIZE = 32

    tt = time.time()
    dataset = HDF5Dataset('codedata_new/data/shuffled_data_0.h5')
    print(len(dataset))
    print('Time taken to load the dataset: ', time.time() - tt)


    tt = time.time()
    bla = dataset[:BATCHSIZE]
    print('Time taken to load first batch: ', time.time() - tt)


    tt = time.time()
    bla = dataset[-BATCHSIZE:]
    print('Time taken to load last batch: ', time.time() - tt)

    tt = time.time()
    random_ind = np.random.randint(0, len(dataset)//BATCHSIZE)
    bla = dataset[random_ind*BATCHSIZE:random_ind*BATCHSIZE+BATCHSIZE]
    print('Time taken to load random batch: ', time.time() - tt)

    tt = time.time()
    inds = np.random.choice(len(dataset), BATCHSIZE)
    inds = np.sort(inds)
    bla = dataset[inds]
    print('Time taken to load batch of random samples: ', time.time() - tt)


    dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False)
    #get the first batch
    for i, (S, X) in enumerate(dataloader):
        print(i, S.shape, X.shape)
        print(f'Time taken to load batch {i}: ', time.time() - tt)
        tt = time.time()
        if i >= 5:
            break

    #get a single batch
    dataloader = HDF5DataLoader(dataset, batch_size=BATCHSIZE, local_rank=2, world_size=8, device='cpu', shuffle=False)
    print(f'Time taken to set up dataloader: ', time.time() - tt)

    tt = time.time()
    dataloader.get_next()
    print(f'Time taken to load first batch: ', time.time() - tt)

    for i in range(5):
        tt = time.time()
        dataloader.get_next()
        print(f'Time taken to load second batch: ', time.time() - tt)

    for i in range(5):
        tt = time.time()
        dataloader.get_batch()
        print(f'Time taken to load random batch: ', time.time() - tt)

    for i in range(5):
        tt = time.time()
        dataloader.get_one()
        print(f'Time taken to load random sample: ', time.time() - tt)