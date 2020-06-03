import torch
import numpy as np
import linecache
from torch.utils.data import Dataset

MAX_NUM_PATHS  = 200

class TextDataset(Dataset):
    
    def __init__(self, dataset_path, value_vocab, path_vocab, target_vocab):
        self.dataset_path = dataset_path
        self.value_vocab = value_vocab
        self.path_vocab = path_vocab
        self.target_vocab = target_vocab
        self.len_dataset = sum(1 for line in open(dataset_path, 'r'))
    
    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self, idx):
        line = linecache.getline(self.dataset_path, idx + 1)
        return self._data_processing_one_line(line)
        
    def _data_processing_one_line(self, line):
        '''
        Processing for the each line of the dataset
        '''
        final_start = []
        final_path = []
        final_ends = []
        final_labels = None
        name, *tree = line.split(' ')
        for each_path in tree:
            if each_path != '' and each_path != '\n':
                temp_path = each_path.split(',')

                final_start.append(self.value_vocab.get(temp_path[0], self.value_vocab['<unk>']))
                final_path.append(self.path_vocab.get(temp_path[1], self.path_vocab['<unk>']))
                final_ends.append(self.value_vocab.get(temp_path[2], self.value_vocab['<unk>']))
                  
        # in order to fulfil to the max number of paths
        final_start += [self.value_vocab['<pad>']] * (MAX_NUM_PATHS - len(final_start))
        final_path += [self.path_vocab['<pad>']] * (MAX_NUM_PATHS - len(final_path))
        final_ends += [self.value_vocab['<pad>']] * (MAX_NUM_PATHS - len(final_ends))
        final_labels = self.target_vocab.get(name, self.target_vocab['<unk>'])
        return torch.tensor(final_start), torch.tensor(final_path), torch.tensor(final_ends), torch.tensor(final_labels)

        


class DatasetBuilder:
    '''
    The purpose of this class is to process a dataset's 
    each line and form 3 tensors of start_words, contexts,
    end_words. Lines are looks like 'method_name start_word, 
    context, end_word'
    '''
    def __init__(self, dataset_path, value_vocab, path_vocab, target_vocab, batch_size = 100):
        self.dataset_path = dataset_path
        self.value_vocab = value_vocab
        self.path_vocab = path_vocab
        self.target_vocab = target_vocab
        self.batch_size = batch_size
        self.len_dataset = 0
    
    def __len__(self):
        return self.len_dataset + 1

    def _data_processing_one_line(self, line):
        '''
        Processing for the each line of the dataset
        '''
        final_start = []
        final_path = []
        final_ends = []
        final_labels = None
        name, *tree = line.split(' ')
        for each_path in tree:
            if each_path != '' and each_path != '\n':
                temp_path = each_path.split(',')

                final_start.append(self.value_vocab.get(temp_path[0], self.value_vocab['<unk>']))
                final_path.append(self.path_vocab.get(temp_path[1], self.path_vocab['<unk>']))
                final_ends.append(self.value_vocab.get(temp_path[2], self.value_vocab['<unk>']))
                  
        # in order to fulfil to the max number of paths
        final_start += [self.value_vocab['<pad>']] * (MAX_NUM_PATHS - len(final_start))
        final_path += [self.path_vocab['<pad>']] * (MAX_NUM_PATHS - len(final_path))
        final_ends += [self.value_vocab['<pad>']] * (MAX_NUM_PATHS - len(final_ends))
        final_labels = self.target_vocab.get(name, self.target_vocab['<unk>'])
        return torch.tensor(final_start), torch.tensor(final_path), torch.tensor(final_ends), torch.tensor(final_labels)

    def _form_tensors(self):
        '''
        This method forms tensors which will be delivered to the model.
        Also, it should be considered as the iterator for the dataset.
        output: tensor_starts, tensor_contexts, tensor_ends, tensor_labels
        '''
        with open(self.dataset_path, 'r') as file:
            import numpy as np

            list_starts = []
            list_contexts = []
            list_ends = []
            list_labels = []
            
            for i, line in enumerate(file):
                self.len_dataset = i
                test_ = self._data_processing_one_line(line)
                list_starts += [test_[0]]
                list_contexts += [test_[1]]
                list_ends += [test_[2]]
                list_labels += [test_[3]]
                
                if (i+1) % self.batch_size == 0:
                    tensor_starts = torch.LongTensor(list_starts)
                    tensor_contexts = torch.LongTensor(list_contexts)
                    tensor_ends = torch.LongTensor(list_ends)
                    tensor_labels = torch.LongTensor(list_labels)

                    list_starts = []
                    list_contexts = []
                    list_ends = []
                    list_labels = []

                    yield tensor_starts, tensor_contexts, tensor_ends, tensor_labels
