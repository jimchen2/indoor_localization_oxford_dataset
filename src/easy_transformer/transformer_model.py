import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pos_embedding(positions)

class IMUTransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, output_size, max_len=500, dropout=0.1):
        super(IMUTransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = LearnedPositionalEncoding(d_model, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.input_linear = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_activation = nn.GELU()
        
        self.output_linear = nn.Linear(d_model, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
        
        self.apply(_init_weights)

    def forward(self, src, return_all_positions=False, pooling='last'):
        # src is shape (batch_size, sequence_length, input_size)
        # example is (128,200,15), won't be much much larger than that
        src = self.input_linear(src)
        src = self.input_norm(src)
        src = self.input_activation(src)
        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        if return_all_positions:
            output = self.dropout(output)
            output = self.output_linear(output)
            # output is shape (batch_size, sequence_length, output_size)
        else:
            if pooling == 'last':
                output = output[:, -1, :]
            elif pooling == 'mean':
                output = output.mean(dim=1)
            else:
                raise ValueError("pooling must be 'last' or 'mean'")
            
            output = self.dropout(output)
            output = self.output_linear(output)
            # output is shape (batch_size, output_size)
        
        return output