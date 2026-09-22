"""Independent capacity settings; original one-layer scan retained at depth one."""
from dataclasses import dataclass
import torch
from torch import nn
from v7_integrated_config import V7IntegratedConfig
from v7_integrated_model import MarketMambaV7Integrated, ResidualMamba2

@dataclass(frozen=True)
class ExperimentConfig(V7IntegratedConfig):
    forward_layers: int = 1
    reverse_layers: int = 1
    def validate(self):
        super().validate()
        if min(self.forward_layers,self.reverse_layers)<1:
            raise ValueError("both scan depths must be positive")
        if self.d_model < 8 or self.d_model % 4:
            raise ValueError("d_model must be >= 8 and divisible by four")

class ExperimentModel(MarketMambaV7Integrated):
    def __init__(self,config,*,ssd_fn=None,graph_layer=None):
        super().__init__(config,ssd_fn=ssd_fn,graph_layer=graph_layer)
        if config.forward_layers>1:
            self.cross_forward=nn.Sequential(self.cross_forward,*(ResidualMamba2(config,ssd_fn) for _ in range(config.forward_layers-1)))
        if config.reverse_layers>1:
            self.cross_reverse=nn.Sequential(self.cross_reverse,*(ResidualMamba2(config,ssd_fn) for _ in range(config.reverse_layers-1)))
    def forward_prepared(self,sample):
        # FastDataset guarantees lexical order, current eligibility and induced graph.
        x=sample["x"]
        temporal=x.new_empty((x.shape[0],self.config.d_model))
        for start, ids in sample["groups"]:
            valid=sample["observation_mask"][ids,start:]
            sequence=self.embedding(torch.where(valid.unsqueeze(-1),x[ids,start:],0.))
            sequence=torch.where(valid.unsqueeze(-1),sequence,self.missing_observation)
            for layer in self.temporal:
                sequence=layer(sequence)
            temporal[ids]=sequence[:,-1]
        graph=self.graph_layer(temporal,sample["edge_index"],sample["edge_attr"])
        seq=temporal.unsqueeze(0)
        forward=self.cross_forward(seq)
        reverse=self.cross_reverse(torch.flip(seq,dims=(1,)))
        scan=(forward+torch.flip(reverse,dims=(1,))).squeeze(0)*.5
        fused,_=self.fusion(temporal,graph,scan)
        fused=self.fused_norm(fused)
        return torch.cat((self.head_5d(fused),self.head_10d(fused)),-1)
