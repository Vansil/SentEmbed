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
        