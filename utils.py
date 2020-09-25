# Install the required libraries
import tensorflow as tf
import unicodedata
import re
import os
import wget
import nltk

# Downloading data from the network to the file system
def load_txt_data():
    print('Downloading dataset')
    # data location address on the network | test
    test_url = 'http://www.hlt.utdallas.edu/~vince/data/emnlp12/test.c.txt'
    # Download the file, after checking its absence in the system
    if not os.path.exists('./test.c.txt'):
        wget.download(test_url, './test.c.txt')

    # data location address on the network | train
    train_url = 'http://www.hlt.utdallas.edu/~vince/data/emnlp12/train.c.txt'
    # Download the file, after checking its absence in the system
    if not os.path.exists('./train.c.txt'):
        wget.download(train_url, './train.c.txt')
        
# loading data from the file system and their preprocessing
def data_preprocessing():
    # test
    test_list_source_target_sents = []
    with open("test.c.txt", "r") as file:
        for line in file:
            test_list_source_target_sents.append(line.split("\n")[0])
    
    test_list_source_target_sents = list(filter(('').__ne__, test_list_source_target_sents))

    # train
    train_list_source_target_sents = []
    with open("train.c.txt", "r") as file:
        for line in file:
            train_list_source_target_sents.append(line.split("\n")[0])
    
    train_list_source_target_sents = list(filter(('').__ne__, train_list_source_target_sents))
    
    # Creation of a general corpus to create the vocabulary required for tokenization
    list_source_target = train_list_source_target_sents + test_list_source_target_sents
    
    return list_source_target

# Create pairs: source / target 
def pairs_source_target(list_source_target_sents):
    list_source_sent = []
    list_target_sent = []
    util_target_list = []
    separator = ', '

    for indx in range(0, len(list_source_target_sents), 4):
        
        list_source_sent.append(list_source_target_sents[indx])
        
        util_target_list.append(list_source_target_sents[indx+1:indx+4])
        util_target_list = separator.join(util_target_list[0])
        
        list_target_sent.append(util_target_list.replace(', ', ',').replace(',', ' '))
        util_target_list = []
        
    return list_source_sent, list_target_sent

# Convert unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # create a space between the word and the punctuation mark
    # example: "He was a student." => "he was a student ."
    # source:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replace everything with spaces except (a-z, A-Z, ".", "?", "!", ",") 
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # add the start and end tokens
    # now the model will know where the prediction starts and ends
    w = '<start> ' + w + ' <end>'
    return w

# create pairs in the format: [source, target]
def create_dataset(source, target):

    new_source = [[preprocess_sentence(s)] for s in source]
    new_target = [[preprocess_sentence(t)] for t in target]
    
    return sum(new_source, []), sum(new_target, [])

# Tokenizing words
def tokenize(seq):
    seq_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    seq_tokenizer.fit_on_texts(seq)

    tensor = seq_tokenizer.texts_to_sequences(seq)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, seq_tokenizer

# create the final data
def load_dataset(source, target):
    
    inp_seq ,targ_seq = create_dataset(source, target)

    input_tensor, inp_seq_tokenizer = tokenize(inp_seq)
    target_tensor, targ_seq_tokenizer = tokenize(targ_seq)

    return input_tensor, target_tensor, inp_seq_tokenizer, targ_seq_tokenizer

# Index to word mapping
def convert(seq, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, seq.index_word[t]))
            
# Converting a tensor to a sentence
def convert_str(seq, tensor):
  empty_str = ''
  for t in tensor:
    if t!=0:
      empty_str += seq.index_word[t] + ' '
  empty_str = empty_str.replace('<start>', '')   
  empty_str = empty_str.replace('<end>', '')
  return empty_str

# Function for assessing quality by BLEU metric on test data
def BLEUscore_test(inp_sequence, targ_sequence, input_tensor, target_tensor):
    accum_score = 0

    for indx in range(len(input_tensor)):

        predicted, _, _ = evaluate(convert_str(inp_sequence, input_tensor[indx]))
        predicted = predicted.replace('<end>', '')
        predicted = predicted.split(' ')
        predicted = list(filter(('').__ne__, predicted))

        targ_str = convert_str(targ_sequence, target_tensor[indx])
        targ_str = targ_str.split(' ')
        targ_str = list(filter(('').__ne__, targ_str))

        accum_score += nltk.translate.bleu_score.sentence_bleu([targ_str], predicted)
        

    BLEUscore = accum_score / len(input_tensor)
    return BLEUscore

