import os, sys
from tools_classifier import DataProcessor, InputExample
import pandas as pd
from IPython import embed

class AtomicProcessor(DataProcessor):
    """Reads in ATOMIC data"""

    @classmethod
    def _read_pkl(cls,input_file):
        df = pd.read_pickle(input_file)
        df = df[df.event.isin(df.sample(100).event.unique())]
        _, lines = zip(*df.iterrows())
        lines = [l.to_dict() for l in lines]
        
        if df.isnull().any().any():
            old_shape = df.shape
            # df = df[~df["inference"].isnull()]
            print("There are nulls in the data", old_shape, df.shape)
            embed();exit()
        return lines
    
    @classmethod
    def _read_csv(cls,input_file):
        df = pd.read_csv(input_file)
        #df = df.sample(min(len(df),1000000))
        _, lines = zip(*df.iterrows())
        lines = [l.to_dict() for l in lines]
        
        if df.isnull().any().any():
            old_shape = df.shape
            # df = df[~df["inference"].isnull()]
            print("There are nulls in the data", old_shape, df.shape)
            embed();exit()
        return lines
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_pred_examples(self, data_dir):
        # return self._create_examples_(
        #     self._read_csv(os.path.join(data_dir, "dev_noLabels.csv")),None,"devNoLabel")
        return self._create_examples_(
            self._read_pkl(os.path.join(data_dir, "data.pkl")),"dim_raw","newData")
    
    def _create_examples_(self, lines, label_col, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["event"]
            text_b = line["inference"]
            if label_col is not None:
                label = line[label_col]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            else:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples
    
    def _example2csv(self, example, keys):
        return [example.__dict__.get(k,"?") for k in keys]
        
class Atomic9dimProcessor(AtomicProcessor):
    def __init__(self):
        self.columns = ["guid","text_a","text_b","label"]
        
    def get_labels(self):
        return ['xNeed', 'xIntent', 'xAttr',
                'xEffect', 'xReact', 'xWant',
                'oEffect', 'oReact', 'oWant']
    
    def _create_examples(self, lines, set_type):
        return self._create_examples_(lines, "dim_raw", set_type)
    
    def example2csv(self, example):
        return self._example2csv(example, self.columns)

class Atomic10dimProcessor(AtomicProcessor):
    def __init__(self):
        self.columns = ["guid","text_a","text_b","label"]
        
    def get_labels(self):
        return ['xNeed', 'xIntent', 'xAttr',
                'xEffect', 'xReact', 'xWant',
                'oEffect', 'oReact', 'oWant', 'unrelated']
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train_negExs.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev_negExs.csv")), "dev")
    def _create_examples(self, lines, set_type):
        return self._create_examples_(lines, "dim", set_type)
    
    def example2csv(self, example):
        return self._example2csv(example, self.columns)
    
class AtomicPrePostProcessor(AtomicProcessor):
    def __init__(self):
        self.columns = ["guid","text_a","text_b","label"]

    def get_labels(self):
        return ["preX", "postX", "postY"]
    
    def _create_examples(self, lines, set_type):
        return self._create_examples_(lines, "dim_prePostXY", set_type)
    
    def example2csv(self, example):
        return self._example2csv(example, self.columns)
    
