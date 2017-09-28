import pandas as pd
import numpy as np

SOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
PAD = '<pad>'

def build_dataset(raw_data, language='en', max_vocab_size=5000):
    if language == 'en':
        import nltk
        tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    elif language == 'jp':
        import tinysegmenter
        tokenizer = tinysegmenter.TinySegmenter()
    else:
        raise ValueError('Language is not supported')
    
    tokens = [tokenizer.tokenize(sentence) for sentence in raw_data]
    
    word_count = {}
    for record in tokens:
        for token in record:
            if token not in word_count:
                word_count[token] = 1
            else:
                word_count[token] += 1
    
    words = sorted(word_count, key=word_count.get, reverse=True)
    
    vocabulary = [PAD, SOS, EOS, UNK] + sorted(words[:4996])
    
    tokens = [[SOS] + record + [EOS] for record in tokens]
    
    sequences = np.array([[vocabulary.index(token) if token in vocabulary else vocabulary.index(UNK) for token in record] for record in tokens])
                        
    return sequences, vocabulary

def load_dataset(
    filename,
    src_lang='jp',
    tgt_lang='en',
    max_src_vocab_size=5000,
    max_tgt_vocab_size=5000,
):
    lang_code = {
        'en' : 'English',
        'jp' : 'Japanese'
    } 
    df = pd.read_excel(filename)
    
    sequences_src, vocab_src = build_dataset(df[lang_code[src_lang]], src_lang, max_src_vocab_size)
    sequences_tgt, vocab_tgt = build_dataset(df[lang_code[tgt_lang]], tgt_lang, max_tgt_vocab_size)
    
    return sequences_src, sequences_tgt, vocab_src, vocab_tgt

def sequence_to_sentence(sequence, vocabulary, language='en'):
    if language=='en':
        join_char = ' '
    elif language=='jp':
        join_char = ''
    else:
        raise ValueError('Language is not supported')
        
    sentence = join_char.join([vocabulary[index] for index in sequence])
    
    return sentence