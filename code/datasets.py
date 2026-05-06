"""CUB-200-2011 dataset loader for AttnGAN training."""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import nltk


class CUBDataset(Dataset):
    """
    CUB-200-2011 dataset with text captions.

    Directory layout expected under cfg.DATA_DIR:
        images/           raw JPEG images
        text/             per-class .txt files with 10 captions each
        train/filenames.pickle
        test/filenames.pickle
        captions.pickle   (optional pre-built dict)

    Returns (image, word_ids, sentence_len, class_id, key)
    """

    def __init__(self, data_dir, split='train', words_num=18,
                 captions_per_image=10, transform=None, low_res=64):
        self.data_dir = data_dir
        self.split = split
        self.words_num = words_num
        self.captions_per_image = captions_per_image
        self.transform = transform
        self.low_res = low_res

        self.filenames, self.class_info = self._load_filenames()
        self.word2idx, self.idx2word, self.n_words = self._build_vocab()
        self.captions = self._load_captions()

        # image sizes for the 3 generator stages
        self.imsize = [64, 128, 256]
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------
    def _build_vocab(self):
        vocab_path = os.path.join(self.data_dir, 'vocab.pickle')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            return vocab['word2idx'], vocab['idx2word'], len(vocab['word2idx'])

        word2idx = {'<pad>': 0, '<eos>': 1}
        caption_dir = os.path.join(self.data_dir, 'text')
        for root, _, files in os.walk(caption_dir):
            for fname in sorted(files):
                if not fname.endswith('.txt'):
                    continue
                with open(os.path.join(root, fname), 'r') as f:
                    for line in f:
                        tokens = nltk.tokenize.word_tokenize(line.strip().lower())
                        for tok in tokens:
                            if tok not in word2idx:
                                word2idx[tok] = len(word2idx)
        idx2word = {v: k for k, v in word2idx.items()}
        with open(vocab_path, 'wb') as f:
            pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
        return word2idx, idx2word, len(word2idx)

    # ------------------------------------------------------------------
    # File lists
    # ------------------------------------------------------------------
    def _load_filenames(self):
        split_path = os.path.join(self.data_dir, self.split, 'filenames.pickle')
        with open(split_path, 'rb') as f:
            filenames = pickle.load(f, encoding='latin1')

        class_info_path = os.path.join(self.data_dir, 'class_info.pickle')
        if os.path.exists(class_info_path):
            with open(class_info_path, 'rb') as f:
                class_info = pickle.load(f, encoding='latin1')
        else:
            class_info = [0] * len(filenames)
        return filenames, class_info

    # ------------------------------------------------------------------
    # Captions
    # ------------------------------------------------------------------
    def _load_captions(self):
        """Load all captions for every file; cache as pickle."""
        cache_path = os.path.join(self.data_dir, f'{self.split}_captions.pickle')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        all_captions = []
        caption_dir = os.path.join(self.data_dir, 'text')
        for fname in self.filenames:
            # fname like '001.Black_footed_Albatross/Black_Footed_Albatross_...'
            cap_file = os.path.join(caption_dir, fname + '.txt')
            caps = []
            with open(cap_file, 'r') as f:
                for line in f:
                    line = line.strip().lower()
                    if line:
                        caps.append(line)
            all_captions.append(caps[:self.captions_per_image])

        with open(cache_path, 'wb') as f:
            pickle.dump(all_captions, f)
        return all_captions

    # ------------------------------------------------------------------
    # Caption → tensor
    # ------------------------------------------------------------------
    def _caption_to_ids(self, caption):
        tokens = nltk.tokenize.word_tokenize(caption)
        word_ids = []
        for tok in tokens:
            if tok in self.word2idx:
                word_ids.append(self.word2idx[tok])
        if len(word_ids) > self.words_num:
            word_ids = word_ids[:self.words_num]
        length = len(word_ids)
        # pad
        while len(word_ids) < self.words_num:
            word_ids.append(0)
        return word_ids, length

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _load_image(self, fname):
        img_path = os.path.join(self.data_dir, 'images', fname + '.jpg')
        img = Image.open(img_path).convert('RGB')

        # Augmentation: resize then random crop
        width, height = img.size
        if self.split == 'train':
            # resize to 304×304, random crop to 256×256
            resize = transforms.Resize(int(self.imsize[-1] * 76 / 64))
            img = resize(img)
            img = transforms.RandomCrop(self.imsize[-1])(img)
            img = transforms.RandomHorizontalFlip()(img)
        else:
            resize = transforms.Resize(int(self.imsize[-1] * 76 / 64))
            img = resize(img)
            img = transforms.CenterCrop(self.imsize[-1])(img)

        # Build multi-scale images [64, 128, 256]
        imgs = []
        for size in self.imsize:
            re_img = transforms.Resize(size)(img)
            imgs.append(self.norm(re_img))
        return imgs

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        class_id = self.class_info[idx]
        imgs = self._load_image(fname)

        # Pick one caption randomly during training, first during eval
        if self.split == 'train':
            cap_idx = np.random.randint(0, len(self.captions[idx]))
        else:
            cap_idx = 0
        caption = self.captions[idx][cap_idx]
        word_ids, length = self._caption_to_ids(caption)

        word_ids = torch.LongTensor(word_ids)
        return imgs, word_ids, length, class_id, fname

    def __len__(self):
        return len(self.filenames)


def collate_fn(batch):
    """Custom collate to stack multi-scale image lists."""
    imgs_list, word_ids, lengths, class_ids, fnames = zip(*batch)
    # imgs_list: list of lists-of-tensors; zip into per-scale batches
    num_scales = len(imgs_list[0])
    imgs_batch = []
    for s in range(num_scales):
        imgs_batch.append(torch.stack([item[s] for item in imgs_list]))

    word_ids = torch.stack(word_ids)
    lengths = torch.LongTensor(lengths)
    class_ids = torch.LongTensor(class_ids)
    return imgs_batch, word_ids, lengths, class_ids, fnames
