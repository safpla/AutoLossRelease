'''
__Author__ == 'Haowen Xu'
__Date__ == '09-10-2018'
Convert all data into json format
'''
import json
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import socket
import os
import sys
import codecs
import re
import random
from nltk.tokenize import word_tokenize

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_dir)
from lib.subword_nmt.subword_nmt.learn_bpe import learn_bpe
from lib.subword_nmt.subword_nmt.apply_bpe import BPE

MT_train_size = 0.8
MT_valid_size = 0.1
MT_test_size = 0.1
POS_train_size = 0.8
POS_valid_size = 0.1
POS_test_size = 0.1
NER_train_size = 0.8
NER_valid_size = 0.1
NER_test_size = 0.1
MAX_SEQ_LEN = 60

def replace_arabic(word):
    word = re.sub(r'[0-9]', '', word)
    if word == '':
        word = '0'
    return word

def token2id(token_id, sent):
    '''
    token_id: dict
    sent: list of words
    '''
    sent_out = [token_id.get(w, 0) for w in sent]
    return sent_out

def subword_gen(infile, outfile, num_symbols):
    infile_stream = codecs.open(infile, encoding='utf-8')
    outfile_stream = codecs.open(outfile, 'w', encoding='utf-8')
    learn_bpe(infile_stream, outfile_stream, num_symbols, is_dict=False, total_symbols=True)

def subword_seg(subword_file, infile, outfile):
    subword_file_stream = codecs.open(subword_file, encoding='utf-8')
    infile_stream = codecs.open(infile, encoding='utf-8')
    outfile_stream = codecs.open(outfile, 'w', encoding='utf-8')
    bpe = BPE(subword_file_stream, -1, '@@', None, None)
    for line in infile_stream:
        outfile_stream.write(bpe.process_line(line))

def get_token_id(token_id_name, corpus=None):
    if not corpus:
        if os.path.exists(token_id_name):
            with open(token_id_name, 'r') as fin:
                token_id = json.load(fin)
                return token_id
        else:
            raise('token_id_file not found')

    tokens = {}
    with open(corpus, 'r') as fin:
        for sent in fin.readlines():
            for word in sent.split():
                if word in tokens:
                    tokens[word] += 1
                else:
                    tokens[word] = 1
    sorted_tokens = sorted(tokens.items(), key=lambda d: d[1], reverse=True)
    tokens_id = {}
    for i, tup in enumerate(sorted_tokens):
        # indexs from 0 to 4 are preserved for other usage
        tokens_id[tup[0]] = i + 5
        json.dump(tokens_id, open(token_id_name, 'w'))
    return tokens_id

def get_postag_dict(root):
    annotation = root[0][1]
    postag_id = {}
    for feature in annotation:
        if 'name' in feature.attrib and feature.attrib['name'] == 'pos':
            count = 5
            for value in feature:
                postag_id[value.attrib['name']] = count
                count += 1
    print('pos_tag_dictionary:')
    print(postag_id)
    return postag_id

def generateMTData():
    data_dir = os.path.join(root_dir, 'Data/mnt/mt')
    de_name = os.path.join(data_dir, 'train.tags.de-en.de')
    en_name = os.path.join(data_dir, 'train.tags.de-en.en')

    process1_de_name = os.path.join(data_dir, 'process1.de')
    process1_en_name = os.path.join(data_dir, 'process1.en')

    subword_de_dict_name = os.path.join(data_dir, 'subword.de.dict')
    subword_en_dict_name = os.path.join(data_dir, 'subword.en.dict')

    process2_de_name = os.path.join(data_dir, 'process2.de')
    process2_en_name = os.path.join(data_dir, 'process2.en')

    de = open(de_name, 'r')
    en = open(en_name, 'r')

    # Preprocess1: Normalization, replacing arabic, tokenization, truncation
    process1_de = open(process1_de_name, 'w')
    process1_en = open(process1_en_name, 'w')

    for de_line, en_line in zip(de.readlines(), en.readlines()):
        if de_line[0] == '<' and de_line[-2] == '>':
            continue
        sample = {}
        sample['de'] = word_tokenize(de_line)
        # TODO: normalization process is removed since case is an important
        # feature in NER task
        #sample['de'] = [w.lower() for w in sample['de']]
        sample['de'] = [replace_arabic(w) for w in sample['de']]

        sample['en'] = word_tokenize(en_line)
        # TODO: normalization process is removed since case is an important
        # feature in NER task
        #sample['en'] = [w.lower() for w in sample['en']]
        sample['en'] = [replace_arabic(w) for w in sample['en']]

        if len(sample['de']) > MAX_SEQ_LEN or len(sample['en']) > MAX_SEQ_LEN:
            continue
        process1_de.write(' '.join(sample['de']) + '\n')
        process1_en.write(' '.join(sample['en']) + '\n')
    process1_de.close()
    process1_en.close()

    # Learning subword
    subword_gen(process1_de_name, subword_de_dict_name, 10000)
    subword_gen(process1_en_name, subword_en_dict_name, 10000)

    # Preprocess2: Applying subword (subword tokenization)
    subword_seg(subword_de_dict_name, process1_de_name, process2_de_name)
    subword_seg(subword_en_dict_name, process1_en_name, process2_en_name)

    # Build subword dictionary
    de_token_id = get_token_id(os.path.join(data_dir, 'de_token_id.json'),
                               corpus=process2_de_name)
    en_token_id = get_token_id(os.path.join(data_dir, 'en_token_id.json'),
                               corpus=process2_en_name)
    de_token_id = get_token_id(os.path.join(data_dir, 'de_token_id.json'))
    en_token_id = get_token_id(os.path.join(data_dir, 'en_token_id.json'))

    # Parallelizing input and target and split datasets
    all_data = []
    de = open(process2_de_name, 'r')
    en = open(process2_en_name, 'r')
    for de_line, en_line in zip(de.readlines(), en.readlines()):
        sample = {}
        sample['input'] = token2id(de_token_id, de_line.split())
        sample['target'] = token2id(en_token_id, en_line.split())
        all_data.append(sample)
    num_samples = len(all_data)
    num_samples_train = int(num_samples * MT_train_size)
    num_samples_valid = int(num_samples * MT_valid_size)
    train_data = all_data[:num_samples_train]
    valid_data = all_data[num_samples_train: num_samples_train + num_samples_valid]
    test_data = all_data[num_samples_train + num_samples_valid:]
    mt_train_file = os.path.join(data_dir, 'mt_train.json')
    mt_valid_file = os.path.join(data_dir, 'mt_valid.json')
    mt_test_file = os.path.join(data_dir, 'mt_test.json')
    json.dump(train_data, open(mt_train_file, 'w'), indent=4)
    json.dump(valid_data, open(mt_valid_file, 'w'), indent=4)
    json.dump(test_data, open(mt_test_file, 'w'), indent=4)

def generatePOSData():
    data_dir = os.path.join(root_dir, 'Data/mnt/pos')
    raw_file_name = os.path.join(data_dir, 'tiger_release_aug07.corrected.16012013.xml')
    tree = ET.ElementTree(file=raw_file_name)
    root = tree.getroot()

    postag_id_name = os.path.join(data_dir, 'postag_id.dict.json')
    word_de_name = os.path.join(data_dir, 'word.de')
    word_pos_name = os.path.join(data_dir, 'word.pos')

    subword_de_name = os.path.join(data_dir, 'subword.de')
    subword_pos_name = os.path.join(data_dir, 'subword.pos')


    # Building pos tag dictionary
    postag_id = get_postag_dict(root)
    json.dump(postag_id, open(postag_id_name, 'w'))

    # XML to text
    body = root[1]
    word_de_stream = open(word_de_name, 'w')
    word_pos_stream = open(word_pos_name, 'w')
    count = 0
    for sent in body.iter(tag='s'):
        sent = sent[0][0]
        de = []
        pos = []
        for word in sent.iter(tag='t'):
            de.append(word.attrib['word'])
            pos.append(word.attrib['pos'])
        # Normalization, replacing arabric and truncation

        # TODO: normalization process is removed since case is an important
        # feature in NER task
        #de = [w.lower() for w in de]
        de = [replace_arabic(w) for w in de]
        if len(de) > MAX_SEQ_LEN:
            continue
        word_de_stream.write(' '.join(de) + '\n')
        word_pos_stream.write(' '.join(pos) + '\n')
        #count += 1
        #if count > 100:
        #    break

    # Using the same subword dictionary in MT task to tokenize word.de
    subword_de_dict_name = os.path.join(root_dir, 'Data/mnt/mt/subword.de.dict')
    subword_seg(subword_de_dict_name, word_de_name, subword_de_name)

    # Split pos tag according to subword segmentation.
    subword_de_stream = open(subword_de_name, 'r')
    subword_pos_stream = open(subword_pos_name, 'w')
    word_pos_stream = open(word_pos_name, 'r')
    for line_de, line_pos in zip(subword_de_stream.readlines(),
                                 word_pos_stream.readlines()):
        line_pos_new = []
        line_de = line_de.split()
        line_pos = line_pos.split()
        point = 0
        try:
            for w in line_de:
                line_pos_new.append(line_pos[point])
                if w[-2:] != '@@':
                    point += 1
        except:
            print(line_de)
            print(line_pos)
            exit()
        subword_pos_stream.write(' '.join(line_pos_new) + '\n')
    subword_de_stream.close()
    subword_pos_stream.close()
    word_pos_stream.close()

    # check length equality
    subword_de_stream = open(subword_de_name, 'r')
    subword_pos_stream = open(subword_pos_name, 'r')
    for lde, lpos in zip(subword_de_stream.readlines(),
                         subword_pos_stream.readlines()):
        lde = lde.split()
        lpos = lpos.split()
        if len(lde) == len(lpos):
            continue
        else:
            print('inequal')
    subword_de_stream.close()
    subword_pos_stream.close()

    # Parallelizing input and target and split datasets
    all_data = []
    subword_de_stream = open(subword_de_name, 'r')
    subword_pos_stream = open(subword_pos_name, 'r')
    de_token_id = get_token_id(os.path.join(root_dir, 'Data/mnt/mt/de_token_id.json'))
    postag_id = postag_id
    for de_line, pos_line in zip(subword_de_stream.readlines(),
                                 subword_pos_stream.readlines()):
        sample = {}
        sample['input'] = token2id(de_token_id, de_line.split())
        sample['target'] = token2id(postag_id, pos_line.split())
        all_data.append(sample)
    num_samples = len(all_data)
    num_samples_train = int(num_samples * POS_train_size)
    num_samples_valid = int(num_samples * POS_valid_size)
    train_data = all_data[:num_samples_train]
    valid_data = all_data[num_samples_train: num_samples_train + num_samples_valid]
    test_data = all_data[num_samples_train + num_samples_valid:]
    pos_train_file = os.path.join(data_dir, 'pos_train.json')
    pos_valid_file = os.path.join(data_dir, 'pos_valid.json')
    pos_test_file = os.path.join(data_dir, 'pos_test.json')
    json.dump(train_data, open(pos_train_file, 'w'), indent=4)
    json.dump(valid_data, open(pos_valid_file, 'w'), indent=4)
    json.dump(test_data, open(pos_test_file, 'w'), indent=4)

def generateNERData():
    data_dir = os.path.join(root_dir, 'Data/mnt/ner')
    raw_train_name = os.path.join(data_dir, 'NER-de-train.tsv')
    raw_valid_name = os.path.join(data_dir, 'NER-de-dev.tsv')
    raw_test_name = os.path.join(data_dir, 'NER-de-test.tsv')

    word_train_de = os.path.join(data_dir, 'word_train.de')
    word_valid_de = os.path.join(data_dir, 'word_valid.de')
    word_test_de = os.path.join(data_dir, 'word_test.de')
    word_train_ner = os.path.join(data_dir, 'word_train.ner')
    word_valid_ner = os.path.join(data_dir, 'word_valid.ner')
    word_test_ner = os.path.join(data_dir, 'word_test.ner')

    subword_train_de = os.path.join(data_dir, 'subword_train.de')
    subword_valid_de = os.path.join(data_dir, 'subword_valid.de')
    subword_test_de = os.path.join(data_dir, 'subword_test.de')
    subword_train_ner = os.path.join(data_dir, 'subword_train.ner')
    subword_valid_ner = os.path.join(data_dir, 'subword_valid.ner')
    subword_test_ner = os.path.join(data_dir, 'subword_test.ner')

    ner_train_file = os.path.join(data_dir, 'ner_train.json')
    ner_valid_file = os.path.join(data_dir, 'ner_valid.json')
    ner_test_file = os.path.join(data_dir, 'ner_test.json')

    nertag_id_name = os.path.join(data_dir, 'nertag_id.dict.json')

    def preprocess1(raw_name, de_name, ner_name):
        raw_stream = open(raw_name, 'r')
        de_stream = open(de_name, 'w')
        ner_stream = open(ner_name, 'w')

        line_de = []
        line_ner = []
        for line in raw_stream.readlines():
            line = line.strip().split('\t')
            # Skip empty line
            if line[0] == '':
                continue
            elif line[0] == '#':
                if len(line_de) > 0:
                    if len(line_de) < MAX_SEQ_LEN:
                        de_stream.write(' '.join(line_de) + '\n')
                        ner_stream.write(' '.join(line_ner) + '\n')
                line_de = []
                line_ner = []
            else:
                line_de.append(line[1])
                line_ner.append(line[2])

        raw_stream.close()
        de_stream.close()
        ner_stream.close()

    # TSV to text
    preprocess1(raw_train_name, word_train_de, word_train_ner)
    preprocess1(raw_valid_name, word_valid_de, word_valid_ner)
    preprocess1(raw_test_name, word_test_de, word_test_ner)

    # Use the same subword dictionary in MT task to tokenize word.de
    subword_de_dict_name = os.path.join(root_dir, 'Data/mnt/mt/subword.de.dict')
    subword_seg(subword_de_dict_name, word_train_de, subword_train_de)
    subword_seg(subword_de_dict_name, word_valid_de, subword_valid_de)
    subword_seg(subword_de_dict_name, word_test_de, subword_test_de)

    # Get ner tag dictionary
    nertag_id = get_token_id(nertag_id_name, corpus=word_train_ner)
    print('ner_tag_dictionary:')
    print(nertag_id)

    # Split ner tag according to subword segmentation
    def preprocess2(subword_de_name, subword_ner_name, word_ner_name):
        subword_de_stream = open(subword_de_name, 'r')
        subword_ner_stream = open(subword_ner_name, 'w')
        word_ner_stream = open(word_ner_name, 'r')
        for line_de, line_ner in zip(subword_de_stream.readlines(),
                                     word_ner_stream.readlines()):
            line_ner_new = []
            line_de = line_de.split()
            line_ner = line_ner.split()
            point = 0
            try:
                for w in line_de:
                    line_ner_new.append(line_ner[point])
                    if w[-2:] != '@@':
                        point += 1
            except:
                print(line_de)
                print(line_ner)
                exit()
            subword_ner_stream.write(' '.join(line_ner_new) + '\n')
        subword_de_stream.close()
        subword_ner_stream.close()
        word_ner_stream.close()

    preprocess2(subword_train_de, subword_train_ner, word_train_ner)
    preprocess2(subword_valid_de, subword_valid_ner, word_valid_ner)
    preprocess2(subword_test_de, subword_test_ner, word_test_ner)

    # Parallelizing input and target.
    def parallelizing(subword_de, subword_ner, ner_file):
        data = []
        subword_de_stream = open(subword_de, 'r')
        subowrd_ner_stream = open(subword_ner, 'r')
        de_token_id = get_token_id(os.path.join(root_dir, 'Data/mnt/mt/de_token_id.json'))
        nertag_id = get_token_id(nertag_id_name)
        for de_line, ner_line in zip(subword_de_stream.readlines(),
                                     subowrd_ner_stream.readlines()):
            sample = {}
            sample['input'] = token2id(de_token_id, de_line.split())
            sample['target'] = token2id(nertag_id, ner_line.split())
            data.append(sample)
        json.dump(data, open(ner_file, 'w'), indent=4)

    parallelizing(subword_train_de, subword_train_ner, ner_train_file)
    parallelizing(subword_valid_de, subword_valid_ner, ner_valid_file)
    parallelizing(subword_test_de, subword_test_ner, ner_test_file)


if __name__ == '__main__':
    generateMTData()
    generatePOSData()
    generateNERData()

