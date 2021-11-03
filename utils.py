import string
import torch
import pandas as pd
from transformers import BertTokenizer
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def words_from_text(s, words_to_ignore=[]):
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    # TODO implement w regex
    words = []
    word = ""
    for c in " ".join(s.split()):
        if c.isalnum():
            word += c
        elif c in "'-" and len(word) > 0:
            # Allow apostrophes and hyphens as long as they don't begin the
            # word.
            word += c
        elif word:
            if word not in words_to_ignore:
                words.append(word)
            word = ""
    if len(word) and (word not in words_to_ignore):
        words.append(word)
    return words


def read_imdb_dataset(path, data):
        df = pd.read_csv(path)
        dataset = []
        
        if data == '500samples_imdb.csv':
            s_list = df["sentence"]
            s_label = df["polarity"]
        elif data == '500samples_mnli.csv':
            s_list = df["sentence"]
            s_label = df["polarity"]
        else:
            s_list = df["text"]
            s_label = df["label"]
        i=0
        tok = BertTokenizer.from_pretrained('bert-base-uncased')
        for index, label in enumerate(s_label):
            sent = s_list[index]
            sent = sent.replace("<br />"," ")
            
            # limit the number of tokens to 510 for bert (since only bert is used for language modeling, see lm_sampling.py)
            sent = words_from_text(sent)
            # print(sent)
            # print(len(sent))
            sent = ' '.join(sent)

           
            tokens = tok.tokenize(sent)
            tokens = tokens[:510]
            text = ' '.join(tokens).replace(' ##', '').replace(' - ', '-').replace(" ' ","'")

            target = int(label)
            i=i+1
            print("第"+str(i)+"条读取中")
            print("text："+str(text)+";target:"+str(target))
            dataset.append((text, target))
                
        return dataset

def write_to_csv(olm_file_name, res_relevances, input_instances, labels_true, labels_pred):
    f = open("olm-files/" + olm_file_name, "w")
    writer = csv.writer(f)  

    writer.writerow(['tokenized_text', 'relevances', 'true label', 'predicted label'])
    
    for i in range(len(res_relevances)):
        list_relevances = []
        for k in range(len(res_relevances[i])):
            list_relevances.append(res_relevances[i][('sent', k)])

        writer.writerow([input_instances[i].sent.tokens, list_relevances, labels_true[i], labels_pred[i]])           
    
    f.close()

def check_if_subword(token, model_type, starting=False):
    """Check if ``token`` is a subword token that is not a standalone word.  检查 ``token`` 是否是不是独立单词的子词标记。

    Args:
        token (str): token to check.
        model_type (str): type of model (options: "bert", "roberta", "xlnet").
        starting (bool): Should be set ``True`` if this token is the starting token of the overall text.
            This matters because models like RoBERTa does not add "Ġ" to beginning token.
    Returns:
        (bool): ``True`` if ``token`` is a subword token.
    """
    avail_models = [
        "bert",
        "gpt",
        "gpt2",
        "roberta",
        "bart",
        "electra",
        "longformer",
        "xlnet",
    ]
    if model_type not in avail_models:
        raise ValueError(
            f"Model type {model_type} is not available. Options are {avail_models}."
        )
    if model_type in ["bert", "electra"]:
        return True if "##" in token else False
    elif model_type in ["gpt", "gpt2", "roberta", "bart", "longformer"]:
        if starting:
            return False
        else:
            return False if token[0] == "Ġ" else True
    elif model_type == "xlnet":
        return False if token[0] == "_" else True
    else:
        return False


def strip_BPE_artifacts(token, model_type):
    """Strip characters such as "Ġ" that are left over from BPE tokenization.

    Args:
        token (str)
        model_type (str): type of model (options: "bert", "roberta", "xlnet")
    """
    avail_models = [
        "bert",
        "gpt",
        "gpt2",
        "roberta",
        "bart",
        "electra",
        "longformer",
        "xlnet",
        "distilroberta",
    ]
    if model_type not in avail_models:
        raise ValueError(
            f"Model type {model_type} is not available. Options are {avail_models}."
        )
    if model_type in ["bert", "electra"]:
        return token.replace("##", "")
    elif model_type in ["gpt", "gpt2", "roberta", "bart", "distilroberta" "longformer"]:
        return token.replace("Ġ", "")
    elif model_type == "xlnet":
        if len(token) > 1 and token[0] == "_":
            return token[1:]
        else:
            return token
    else:
        return token


def check_if_punctuations(word):
    """Returns ``True`` if ``word`` is just a sequence of punctuations."""
    for c in word:
        if c not in string.punctuation:
            return False
    return True

def is_one_word(word):
    return len(words_from_text(word)) == 1