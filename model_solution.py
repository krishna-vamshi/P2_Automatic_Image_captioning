import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size,embed_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = 1, batch_first = True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        emb = self.embed_layer(captions)
        emb = torch.cat((features.unsqueeze(1), emb), dim = 1)
        lstm_out, _ = self.lstm(emb)
        
        out = self.fc(lstm_out)
        
        return out
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        out_sent = []
        
        for i in range(max_len):
            
            lstm_out, states = self.lstm(inputs, states)
            lstm_out = self.fc(lstm_out)
        
            out = lstm_out.squeeze(1)
            word_next = out.max(1)[1]
            out_sent.append(word_next.item())
            inputs = self.embed_layer(word_next).unsqueeze(1)
            
        
        return out_sent