import pandas as pd
from dont_patronize_me import DontPatronizeMe
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from autocorrect import Speller
import contractions
import string
import re
from nlpaug.augmenter.word import ContextualWordEmbsAug, SynonymAug
from tqdm import tqdm
from nltk.corpus import stopwords

class DataProcessor:
    def __init__(self) -> None:

        self.raw_data = self.load_data(is_train=True)
        self.trdf = self.get_data(is_train=True)
        self.tedf = self.get_data(is_train=False)
        self.test_data = self.load_data(is_train=False)
    
    def load_data(self, is_train:bool=True) -> pd.DataFrame:
        
        if is_train:
            rows=[]
            with open('./Data/dontpatronizeme_pcl.tsv') as f:
                for line in f.readlines()[4:]:
                    par_id=line.strip().split('\t')[0]
                    art_id = line.strip().split('\t')[1]
                    keyword=line.strip().split('\t')[2]
                    country=line.strip().split('\t')[3]
                    t=line.strip().split('\t')[4]#.lower()
                    l=line.strip().split('\t')[-1]
                    if l=='0' or l=='1':
                        lbin=0
                    else:
                        lbin=1

                    rows.append({'par_id':par_id,
                        'art_id':art_id,
                        'keyword':keyword,
                        'country':country,
                        'text':t, 
                        'label':lbin, 
                        'orig_label':l
                        })
        else:
            rows = []
            with open('./Data/task4_test.tsv') as f:
                for line in f.readlines()[0:]:
                    par_id=line.strip().split('\t')[0]
                    art_id = line.strip().split('\t')[1]
                    keyword=line.strip().split('\t')[2]
                    country=line.strip().split('\t')[3]
                    t=line.strip().split('\t')[4]#.lower()
                    l=line.strip().split('\t')[-1]
                    if l=='0' or l=='1':
                        lbin=0
                    else:
                        lbin=1

                    rows.append({'par_id':par_id,
                        'art_id':art_id,
                        'keyword':keyword,
                        'country':country,
                        'text':t, 
                        'label':lbin, 
                        'orig_label':l
                        })
                
        df=pd.DataFrame(rows, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label'])
        return df
            
    def get_data(self, is_train) -> pd.DataFrame:
        if is_train:
            ids = pd.read_csv('./Data/train_semeval_parids-labels.csv')
            ids.par_id = ids.par_id.astype(str)
            
        else:
            ids = pd.read_csv('./Data/dev_semeval_parids-labels.csv')
            ids.par_id = ids.par_id.astype(str)
        
        rows = []
        for idx in range(len(ids)):
            parid = ids.par_id[idx]

            # select row from original dataset to retrieve `text` and binary label
            keyword = self.raw_data.loc[self.raw_data.par_id == parid].keyword.values[0]
            text = self.raw_data.loc[self.raw_data.par_id == parid].text.values[0]
            label = self.raw_data.loc[self.raw_data.par_id == parid].label.values[0]
            original_label = self.raw_data.loc[self.raw_data.par_id == parid].orig_label.values[0]
            rows.append({
                'par_id':parid,
                'keyword':keyword,
                'text':text,
                'label':label,
                "orig_label":original_label
            })
            
        return pd.DataFrame(rows)
    
    def preprocess(self, data:pd.DataFrame, is_lower:bool=False, is_correct_spelling:bool=False, 
                   is_expand_contraction:bool=False, is_remove_punctuation:bool=False, is_remove_stopwords:bool=False) -> pd.DataFrame:
        
        # Drop NaN values
        data = data.dropna()
        
        # Clean text
        tqdm.pandas(desc="Clean Text Processing")
        data.text = data.text.progress_apply(lambda x: self._clean_text(x))
        
        # Lowercase
        if is_lower:
            tqdm.pandas(desc="Lowering Text Processing")
            data.text = data.text.progress_apply(lambda x: x.lower())
        
        # Spell correction
        if is_correct_spelling:
            tqdm.pandas(desc="Spell Correction Processing")
            data.text = data.text.progress_apply(lambda x: self._spell_correction(x))
        
        # Expand contractions
        if is_expand_contraction:
            tqdm.pandas(desc="Expand Contractions Processing")
            data.text = data.text.progress_apply(lambda x: self._expand_contractions(x))
        
        # Remove punctuation
        if is_remove_punctuation:
            tqdm.pandas(desc="Remove Punctuation Processing")
            data.text = data.text.progress_apply(lambda x: self._remove_punctuation(x))
        
        # Remove stopwords
        if is_remove_stopwords:
            tqdm.pandas(desc="Remove Stopwords Processing")
            data.text = data.text.progress_apply(lambda x: self._remove_stopwords(x))
            
        return data
    
    def _clean_text(self, text: str) -> str:
        text = text.strip('"')
        text = text.replace("\\'", "'")
        text = text.replace(" '", "'")
        
        text = re.sub(r'n\'t', "not", text) # replace n't with not
        return text
    
    def _spell_correction(self, text: str) -> str:
        speller = Speller(lang="en")
        return " ".join(speller(word) for word in text.split())
    
    def _expand_contractions(self, text: str) -> str:
        return contractions.fix(text)
    
    def _remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def _remove_stopwords(self, text: str) -> str:
        return " ".join([word for word in text if word not in stopwords.words("english") and word.isalpha()])
    
    def data_augmentation(self, data:pd.DataFrame, is_contextual:bool=False, is_synonym:bool=False, 
                          is_raw_upsampling:bool=False, is_raw_downsampling:bool=False, augmentation_repeat:int=0, device:str="cpu") -> pd.DataFrame:
        data = data.copy()
        
        if is_contextual:
            contextual_augmented_data = self._contextual_augmentation(data, augmentation_repeat, device=device)
        else:
            contextual_augmented_data = pd.DataFrame()
        
        if is_synonym:
            synonym_augmented_data = self._synonym_augmentation(data, augmentation_repeat)
        else:
            synonym_augmented_data = pd.DataFrame()
        
        if is_raw_upsampling:
            raw_upsampled_data = self._raw_upsampling(data)
        else:
            raw_upsampled_data = pd.DataFrame()
        
        if is_raw_downsampling:
            pcldf = data[data.label==1]
            npos = len(pcldf)
            raw_downsampled_data = pd.concat([pcldf,data[data.label==0][:npos*2]])
            
            return pd.concat([raw_downsampled_data, contextual_augmented_data, synonym_augmented_data], ignore_index=True)
        
        return pd.concat([data, contextual_augmented_data, synonym_augmented_data, raw_upsampled_data], ignore_index=True)
    
    def _raw_upsampling(self, data:pd.DataFrame):
        data = data.copy()
        
        negative_data = data[data.label==0]
        positive_data = data[data.label==1]
        
        positive_data = positive_data.sample(frac=len(negative_data)/(3*len(positive_data)), random_state=42, replace=True).reset_index(drop=True) ## added replace=True -Kyo
        
        return positive_data

    def _contextual_augmentation(self, data:pd.DataFrame, augmentation_repeat:int, device:str):

        keywords = list(data.keyword.unique())
        
        augmenter = ContextualWordEmbsAug(model_path = 'distilbert-base-cased', device=device,
                                    action="substitute", top_k=30, stopwords=keywords)
        
        positive_data = data[data.label==1]
        
        if augmentation_repeat == 0:
            positive_data.text = positive_data.text.apply(lambda x: augmenter.augment(x))
            return positive_data
        else:
            augmented_positive_data = pd.DataFrame()

            for i in range(augmentation_repeat):
                positive_data_copy = positive_data.copy()
                
                tqdm.pandas(desc="Contextual Augmentation Processing")
                positive_data_copy.text = positive_data_copy.text.progress_apply(lambda x: "".join(augmenter.augment(x)))

                augmented_positive_data = pd.concat([augmented_positive_data, positive_data_copy], ignore_index=True)

            return augmented_positive_data
    
    def _synonym_augmentation(self, data:pd.DataFrame, augmentation_repeat:int):
        keywords = list(data.keyword.unique())
        augmenter = SynonymAug(aug_src='wordnet', stopwords=keywords)
        
        positive_data = data[data.label==1]
        
        if augmentation_repeat == 0:
            positive_data.text = positive_data.text.apply(lambda x: augmenter.augment(x))
            return positive_data
        else:
            augmented_positive_data = pd.DataFrame()
            for i in range(augmentation_repeat):
                positive_data_copy = positive_data.copy()
                tqdm.pandas(desc="Synonym Augmentation Processing")
                positive_data_copy.text = positive_data_copy.text.progress_apply(lambda x: "".join(augmenter.augment(x)))
            
                augmented_positive_data = pd.concat([augmented_positive_data, positive_data_copy], ignore_index=True)

            return augmented_positive_data
        
    def get_dataloader(self, tokenized_text, label, batch_size, shuffle=False):
        return DataLoader(PatronizingDataset(tokenized_text, label), batch_size=batch_size, shuffle=shuffle)


class PatronizingDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return {'text': text, 'label': label}

    
if __name__ == "__main__":
    pass
