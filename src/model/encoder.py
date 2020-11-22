import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class BERTRPMEncoder(BaseModel):
    def __init__(self, args):
        super(BERTRPMEncoder, self).__init__()
        if args.encoder_dropout > 0.:
            self.sfc_dropout = nn.Dropout(args.encoder_dropout)
            self.vfc_dropout = nn.Dropout(args.encoder_dropout)

        self.sfc1 = nn.Linear(300, 300)  # Take the prior Random Projection as one Reduction Layer

        self.vfc1 = nn.Linear(300, 300)
        self.vfc2 = nn.Linear(300, 300)

    def forward(self, x, y):
        """ sentence branch """
        if hasattr(self, 'stc_dropout'):
            x = self.sfc_dropout(x)
            x = F.relu_(self.sfc1(x))
        else:
            x = F.relu_(self.sfc1(x))

        """ video branch """
        if hasattr(self, 'vfc_dropout'):
            y = self.vfc_dropout(y)
            y = self.vfc_dropout(F.relu_(self.vfc1(y)))
            y = F.relu_(self.vfc2(y))
        else:
            y = F.relu_(self.vfc1(y))
            y = F.relu_(self.vfc2(y))

        return x, y


class BERTEncoder(BaseModel):
    def __init__(self, args):
        super(BERTEncoder, self).__init__()
        if args.encoder_dropout > 0.:
            self.sfc_dropout = nn.Dropout(args.encoder_dropout)
            self.vfc_dropout = nn.Dropout(args.encoder_dropout)

        self.sfc1 = nn.Linear(768, 300, bias=False)
        self.sfc2 = nn.Linear(300, 300)

        self.vfc1 = nn.Linear(2048, 300, bias=False)
        self.vfc2 = nn.Linear(300, 300)
        self.vfc3 = nn.Linear(300, 300)

    def forward(self, x, y):
        """ sentence branch """
        if hasattr(self, 'stc_dropout'):
            x = self.sfc_dropout(x)
            x = self.sfc_dropout(F.relu_(self.sfc1(x)))
            x = F.relu_(self.sfc2(x))
        else:
            x = F.relu_(self.sfc1(x))
            x = F.relu_(self.sfc2(x))

        """ video branch """
        if hasattr(self, 'vfc_dropout'):
            y = self.vfc_dropout(y)
            y = self.vfc_dropout(F.relu_(self.vfc1(y)))
            y = self.vfc_dropout(F.relu_(self.vfc2(y)))
            y = F.relu_(self.vfc3(y))
        else:
            y = F.relu_(self.vfc1(y))
            y = F.relu_(self.vfc2(y))
            y = F.relu_(self.vfc3(y))

        return x, y


class LXMERTEncoder(BaseModel):
    def __init__(self, args):
        super(LXMERTEncoder, self).__init__()
        if args.encoder_dropout > 0.:
            self.sfc_dropout = nn.Dropout(args.encoder_dropout)
            self.vfc_dropout = nn.Dropout(args.encoder_dropout)

        self.sfc1 = nn.Linear(768, 300, bias=False)
        self.sfc2 = nn.Linear(300, 300)

        self.vfc1 = nn.Linear(768, 300, bias=False)
        self.vfc2 = nn.Linear(300, 300)
        self.vfc3 = nn.Linear(300, 300)

    def forward(self, x, y):
        """ sentence branch """
        if hasattr(self, 'stc_dropout'):
            x = self.sfc_dropout(x)
            x = self.sfc_dropout(F.relu_(self.sfc1(x)))
            x = F.relu_(self.sfc2(x))
        else:
            x = F.relu_(self.sfc1(x))
            x = F.relu_(self.sfc2(x))

        """ video branch """
        if hasattr(self, 'vfc_dropout'):
            y = self.vfc_dropout(y)
            y = self.vfc_dropout(F.relu_(self.vfc1(y)))
            y = self.vfc_dropout(F.relu_(self.vfc2(y)))
            y = F.relu_(self.vfc3(y))
        else:
            y = F.relu_(self.vfc1(y))
            y = F.relu_(self.vfc2(y))
            y = F.relu_(self.vfc3(y))

        return x, y


