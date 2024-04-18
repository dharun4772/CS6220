#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 07:39:30 2024

@author: taiwei
"""

import pandas as pd
#return the sequentized data and vocabulary of all texts
def get_tokenized_data(read_from,save,file="airline_review_tokenized.csv"):
    import ast
    review_data=pd.read_csv(read_from)
    texts=list(review_data['content'])

    import nltk
    import re
    from nltk.corpus import words, stopwords,wordnet
    from nltk.stem import WordNetLemmatizer
    print("downloading nltk packages......")
    nltk.download('words')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('maxent_ne_chunker')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    print("done")

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN                                                   # By default, assume noun

    english_words = set(words.words())
    def preprocessing_text(text):
        
        lemmatizer = WordNetLemmatizer()
        cleaned_text = re.sub(r'<.*?>', '', text)                                 # Remove HTML
        #cleaned_text = re.sub(r'\d+', '', cleaned_text)                           # Remove numbers
        words = nltk. word_tokenize(cleaned_text)
        no_nonEnglish = [word.lower() for word in words if word.lower() in english_words]            # get rid off non_english words
        no_stopwords = [word for word in no_nonEnglish if word not in stopwords.words('english')]          # get rid off stopwords
        tagged = nltk.pos_tag(no_stopwords)                                                          # tag the words to enhence the performance of lemmatizer
        lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]            # lemmatizing
        if(lemmatized_words==[]):                                            # for empty text line, change it into 'flight' because torch.nn.module expect non empty input for model
            return "flight"
        return ' '.join(lemmatized_words)  

    review_data=review_data[['header','content','rating','recommended']]                # concerning columns of the data
    print("preprocessing tokens......")
    review_data['content']=review_data['content'].apply(preprocessing_text)     
    review_data['header']=review_data['header'].apply(preprocessing_text)
    
    texts=list(review_data['content'])
    texts+=list(review_data['header'])                                                 # our vocabulary build on both reviews and headers
    print("done")

    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    tokenizer = get_tokenizer("basic_english")
    def yield_tokens(data):
        for text in data:
            yield tokenizer(text)
    vocab = build_vocab_from_iterator(yield_tokens(texts), specials=["<unk>"])           # build up the vocab
    vocab.set_default_index(vocab["<unk>"])
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    review_data['token_review']=review_data['content'].apply(text_pipeline)
    review_data['token_header']=review_data['header'].apply(text_pipeline)

    if save==True:
        review_data.to_csv(file)
    return review_data, vocab

def import_tokenized_data(read_from='airline_review_tokenized.csv'):                   # if we already have the tokenized data, just import it and build a vocabulary
    import ast
    print("loading tokenized reviews....")
    token_data=pd.read_csv(read_from)
    token_data['token']=token_data['token'].apply(ast.literal_eval)
    print("done")
    texts=list(token_data['content'])
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    tokenizer = get_tokenizer("basic_english")
    def yield_tokens(data):
        for text in data:
            yield tokenizer(text)
    vocab = build_vocab_from_iterator(yield_tokens(texts), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return token_data, vocab

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

def collate_fn(batch):                                                                        # building dataloader from the dataset
    sequences1, sequences2, labels = zip(*batch)                                              # corresponds to the reviews
    input1=[seq for seq in sequences1]                                                        # corresponds to the headers                                                      
    input2=[seq for seq in sequences2]
    lengths1 = torch.tensor([len(seq) for seq in input1])                                     # corresponding lengths
    lengths2 = torch.tensor([len(seq) for seq in input2])

    input1_padded = pad_sequence(input1, batch_first=True, padding_value=0)                   # padded inputs, padded on each batch
    input2_padded = pad_sequence(input2, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float)
    return input1_padded,input2_padded, labels,lengths1,lengths2

class MyDataset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = data1                                                                    # reviews
        self.data2 = data2                                                                    # headers
        self.labels = labels

    def __len__(self):
        if len(self.data1)!=len(self.data2):                                                  # each row of data has review and header, there number are assumed to be the same
            print("expecting same length of two channels")
            return -1
        return len(self.data1)

    def __getitem__(self, idx):
        sequence_tensor1 = torch.tensor(self.data1[idx], dtype=torch.long)  # Ensure the data type is appropriate (e.g., torch.long for token indices)
        sequence_tensor2 = torch.tensor(self.data2[idx], dtype=torch.long)  # Ensure the data type is appropriate (e.g., torch.long for token indices)

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)  # Adjust dtype based on what's needed for your model/loss function
        return sequence_tensor1, sequence_tensor2, label_tensor

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
def get_embedding_matrix(vocab):                                               # get the embedding matrix for the words, from Stanford GloVe
    print("building embedding dictionary....")
    embeddings_dict = {}
    with open('glove.840B.300d.txt', 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            try:
                vector = np.asarray([i if i!='.' else '0' for i in values[1:]], dtype='float32')
            except:
                continue;
            embeddings_dict[word] = vector
    print("done")



    embedding_dim=300
    vocab_size=len(vocab)+1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    miss_word=[]


    # Creating the embedding matrix
    print("building pretrained word embedding....")
    stddev=np.sqrt(2./embedding_dim)
    for word, i in vocab.get_stoi().items():
        if word in embeddings_dict:
            embedding_matrix[i] = embeddings_dict.get(word)
        else:
            miss_word.append(word)
            #embedding_matrix[i] = np.random.normal(0,stddev,size=(1,embedding_dim))   # something called he normalization
    print("done")
    print("miss word: ", miss_word)
    return embedding_matrix, miss_word



class TextModel1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix):                  # model that concentrate on both review and header
        super(TextModel1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)  # Set pre-trained weights
        self.embedding.requires_grad = False
        self.lstm1 = nn.LSTM(embedding_dim, 50, batch_first=True)
        self.lstm2 = nn.LSTM(50, 10, batch_first=True)
        self.lstm3 = nn.LSTM(embedding_dim, 10, batch_first=True)
        
        
        self.dropout1=nn.Dropout(0.2)
        self.fc1 = nn.Linear(20, 5)  
        self.dropout=nn.Dropout(0.2)
        self.fc2 = nn.Linear(5, 1)   

    def forward(self, x1, x2, lengths1, lengths2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x1 = nn.utils.rnn.pack_padded_sequence(x1, lengths1, batch_first=True, enforce_sorted=False)
        x2 = nn.utils.rnn.pack_padded_sequence(x2, lengths2, batch_first=True, enforce_sorted=False)
        lstm1_output, _ = self.lstm1(x1)
        lstm2_output, (hidden1,_) = self.lstm2(lstm1_output)
        lstm3_output, (hidden2,_) = self.lstm3(x2)
        lstm_output=torch.cat((hidden1[-1], hidden2[-1]),dim=1)
          
        x=self.dropout1(lstm_output)
        x = self.fc1(x)
        x = self.dropout(x)
        x = torch.relu(x)  
        x = self.fc2(x)
        return x


import numpy as np
def get_data_loaders(X1, X2 ,Y, batch_size, random_state_train_test=114, random_state_train_valid=514):                   # X1 are reviews, and X2 are headers
    print("establishing train_test data.....")
    

    from sklearn.model_selection import train_test_split
    train_ft1,test_ft1,train_lbl,test_lbl = train_test_split(X1,Y,test_size=0.2,random_state=random_state_train_test)
    train_ft1,valid_ft1,train_lbl,valid_lbl=train_test_split(train_ft1,train_lbl,test_size=0.1,random_state=random_state_train_valid)
    train_ft2,test_ft2,train_lbl,test_lbl = train_test_split(X2,Y,test_size=0.2,random_state=random_state_train_test)
    train_ft2,valid_ft2,train_lbl,valid_lbl=train_test_split(train_ft2,train_lbl,test_size=0.1,random_state=random_state_train_valid)
    
    
    train_dataset = MyDataset(train_ft1,train_ft2,train_lbl)
    test_dataset = MyDataset(test_ft1,test_ft2,test_lbl)
    valid_dataset = MyDataset(valid_ft1,valid_ft2,valid_lbl)
    train_loader = DataLoader(train_dataset, batch_size=batch_size[0], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,batch_size=batch_size[1],  collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size[2],collate_fn=collate_fn)
    print("done")
    return train_loader, test_loader, valid_loader


def train_model(train_loader,test_loader, valid_loader,model,criterion, optimizer, num_epochs=5):
    print(model)
    print("training...")
    train_loss=[]
    valid_loss=[]
    acc=0
    for epoch in range(num_epochs):
        model.train()
        correct=0
        total=0
        for inputs1,inputs2, labels, lengths1,lengths2 in valid_loader:
            inputs1,inputs2,labels=inputs1.to(device),inputs2.to(device),labels.to(device)
            outputs=model(inputs1,inputs2,lengths1.cpu(), lengths2.cpu())
            
            optimizer.zero_grad()
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            model.eval()
            with torch.no_grad():
                total_loss=0
                total=0
                correct=0
                for inputs1,inputs2, labels, lengths1,lengths2 in valid_loader:
                    inputs1,inputs2,labels=inputs1.to(device),inputs2.to(device),labels.to(device)
                    outputs=model(inputs1,inputs2,lengths1.cpu(), lengths2.cpu())
                    loss=criterion(outputs.squeeze(), labels)
                    total_loss+=loss.item()*len(labels)
                    predicts=(outputs>0.5).float().squeeze()
                    correct+=(predicts==labels).float().sum()
                    total+=len(labels)
                valid_loss.append(total_loss/total)
                acc=correct/total
            model.train()
        print(f'Epoch {epoch+1},  val_acc: {acc}')
        
    print("done")
    
    
    import matplotlib.pyplot as plt
    plt.plot(valid_loss, color='r', label='valid')
    plt.plot(train_loss, color='b', label='train')
    plt.legend(title='Topics')
    plt.show()
    

def save_model(model, dummy_input):
    torch.save(model.state_dict(),"airlint_review.pth") #save the model parameters to a .pth file for python to read, load or fine tune
    
def evaluate_model(train_loader, test_loader, model, criterion):
    print(model)
    correct = 0
    total = 0
    print("evaluating.....")
    prediction=[]
    answers=[]
    with torch.no_grad():
        for inputs1,inputs2, labels, lengths1,lengths2 in test_loader:
            inputs1,inputs2,labels=inputs1.to(device),inputs2.to(device),labels.to(device)
            outputs = model(inputs1,inputs2,lengths1.cpu(), lengths2.cpu())
            predicts=(outputs>0.5).float().squeeze()
            correct+=(predicts==labels).float().sum()
            prediction.append(list(predicts.cpu().float().squeeze().data))
            answers.append(labels.cpu().float().squeeze())
            total+=len(labels)
    #precision: tp/p
    #recall: tp/t
    accuracy=correct/total
    print(total)
    
    answers=np.array(answers[:-1]).flatten()
    prediction=np.array(prediction[:-1]).flatten()
    hit=sum(1.*((answers==1)&(prediction==1)))
    precision = hit / prediction.sum()
    recall = hit / answers.sum()
    f1_score=2*precision*recall/(precision+recall)
    print(f'Accuracy of the network on the testing datapoints: {100 * correct / total} %')
    print(f'F1_score is: {f1_score}')
    print("done")
    
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns


    cm = confusion_matrix(answers, prediction)

    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return accuracy,f1_score,recall,precision


def build_train_model1(embedding_matrix,vocab_size, embedding_dim, device,epoch=10):
    X1=list(token_data['token_review']) 
    X2=list(token_data['token_header'])
    Y=np.array((token_data['recommended']=='yes')*1.)
    train_loader,test_loader,valid_loader=get_data_loaders(X1, X2, Y,[512,32,64],114514,1919810)   # [train,test,valid] batch_size, last 2 parameters are random status for 
    for inputs1,inputs2,_,lengths1,lengths2 in train_loader:
        dummy_input=(inputs1,inputs2,lengths1,lengths2)
        break

    model = TextModel1(vocab_size, embedding_dim, torch.tensor(embedding_matrix, dtype=torch.float)).to(device)
    #describe_model(model, dummy_input)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam( [param for param in model.parameters() if param.requires_grad])         # Adam, using default learning rate
    train_model(train_loader, test_loader, valid_loader ,model, criterion, optimizer,epoch)

    evaluate_model(train_loader, test_loader, model, criterion)
    save_model(model, dummy_input)
    

    
import subprocess
def describe_model(model, dummy_input):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('runs/model_experiment_1')

    writer.add_graph(model, dummy_input)
    writer.close()
    result=subprocess.run(['tensorboard', '--logdir=runs'])

def build_train_model1_reg(embedding_matrix,vocab_size, embedding_dim, device,epoch=10):
    X1=list(token_data['token_review']) 
    X2=list(token_data['token_header'])
    Y=np.array((token_data['recommended']=='yes')*1.)
    

if __name__=="__main__":
    
    try:
        result=subprocess.run(['unzip','results.zip'])  #run c++ code filter.cpp to get the filtered images
        result=subprocess.run(['unzip','dictionaries.zip'])
    except:
        print("may be already unzipped,or not exists")
    token_data, vocab=get_tokenized_data('full_airline_review.csv',False)
    vocab_size=len(vocab)+1
    embedding_dim=300
    #token_data=import_tokenized_data('airline_review_tokenized.csv')
    embedding_matrix,_=get_embedding_matrix(vocab)
    device="cuda" if torch.cuda.is_available() else "cpu" 
    build_train_model1(embedding_matrix, vocab_size, embedding_dim, device,epoch=15)   