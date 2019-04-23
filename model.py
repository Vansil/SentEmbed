import torch.nn as nn
import torch



class Combine(nn.Module):
    '''
    Layer that takes two sentence embeddings and combines them
    Concatenates: sentence embeddings, absolute difference, element-wise product
    '''
    def forward(self, premise, hypothesis):
        absdiff = abs(premise-hypothesis)
        product = premise * hypothesis
        out = torch.cat((premise, hypothesis, absdiff, product), dim=1)

        return out


class Average(nn.Module):
    '''
    Layer that averages its input in dimension 1
    Used for the Baseline sentence embedding
    '''
    def forward(self, x):
        return x.mean(dim=1)

class UniLstmNet(nn.Module):
    '''
    Unidirectional LSTM network, the last hidden state is the sentence representation
    '''
    def __init__(self, embed_weights):
        super(UniLstmNet,self).__init__()

        hidden_dim = 300
        self.embed = nn.Embedding.from_pretrained(embed_weights)
        self.lstm = nn.LSTM(self.embed.embedding_dim, hidden_dim)

        self.comb  = Combine()

        self.sequential = nn.Sequential(
            nn.Linear(1200,512),
            nn.ReLU(),
            nn.Linear(512,3)
        )
        
    def forward(self, premise, hypothesis):
        embed_premise = self.encode(premise)
        embed_hypothesis = self.encode(hypothesis)

        # Classifier
        combined = self.comb(embed_premise, embed_hypothesis)
        out = self.sequential(combined)

        return out

    def encode(self, sentence):
        '''
        Encode a sentence for SentEval evaluation
        '''
        # GloVe embedding
        embed = self.embed(sentence)
        embed = embed.permute([1,0,2])

        # LSTM
        hidden_dim = 300
        batch_size = sentence.shape[0]
        hidden_0 = torch.zeros(1, batch_size, hidden_dim)
        state_0  = torch.zeros(1, batch_size, hidden_dim)
        _, (hidden, _) = self.lstm(embed, (hidden_0, state_0))

        return hidden.squeeze()
    
    def trainable_params(self):
        return [p for p in self.lstm.parameters()] + [p for p in self.sequential.parameters()]
        


class BaselineNet(nn.Module):
    '''
    Baseline network with one hidden layer of 100 neurons
    '''
    def __init__(self, embed_weights):
        super(BaselineNet,self).__init__()

        self.embed = nn.Embedding.from_pretrained(embed_weights)
        self.avg   = Average()
        self.comb  = Combine()

        self.sequential = nn.Sequential(
            nn.Linear(1200,512),
            nn.ReLU(),
            nn.Linear(512,3)
        )
        
    def forward(self, premise, hypothesis):
        embed_premise    = self.avg(self.embed(premise))
        embed_hypothesis = self.avg(self.embed(hypothesis))

        combined = self.comb(embed_premise, embed_hypothesis)
        out = self.sequential(combined)

        return out

    def encode(self, sentence):
        '''
        Encode a sentence for SentEval evaluation
        '''
        return self.avg(self.embed(sentence))

    def trainable_params(self):
        return [p for p in self.sequential.parameters()]
        