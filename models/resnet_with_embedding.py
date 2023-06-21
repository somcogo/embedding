import torch
from torch import nn

from .edmcode import UNetBlock, FeedForward

class ResNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes, in_channels=3, embed_dim=2, layers=[3, 4, 6, 3], site_number=1, use_hypnns=False, version=None, lightweight=None):
        super().__init__()

        self.embedding = nn.Embedding(site_number, embedding_dim=embed_dim)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ffwrd0 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=3*3, version=version) if lightweight else None
        self.ffwrd1 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=3*3, version=version) if lightweight else None
        self.ffwrd2 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=3*3, version=version) if lightweight else None
        self.ffwrd3 = FeedForward(in_channels=embed_dim, hidden_layer=64, out_channels=3*3, version=version) if lightweight else None

        self.layer0 = self._make_layer(layers[0], in_channels=64, out_channels=64, embed_dim=embed_dim, use_hypnns=use_hypnns, version=version, lightweight=lightweight, ffwrd=self.ffwrd0)
        self.layer1 = self._make_layer(layers[1], in_channels=64, out_channels=128, embed_dim=embed_dim, use_hypnns=use_hypnns, version=version, lightweight=lightweight, ffwrd=self.ffwrd1)
        self.layer2 = self._make_layer(layers[2], in_channels=128, out_channels=256, embed_dim=embed_dim, use_hypnns=use_hypnns, version=version, lightweight=lightweight, ffwrd=self.ffwrd2)
        self.layer3 = self._make_layer(layers[3], in_channels=256, out_channels=512, embed_dim=embed_dim, use_hypnns=use_hypnns, version=version, lightweight=lightweight, ffwrd=self.ffwrd3)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, depth, in_channels, out_channels, embed_dim, use_hypnns=False, version=None, lightweight=None, ffwrd=None):
        
        blocks = nn.ModuleList()
        blocks.append(UNetBlock(in_channels=in_channels, out_channels=out_channels, emb_channels=embed_dim, use_hypnns=use_hypnns, version=version, ffwrd=ffwrd, lightweight=lightweight))

        for _ in range(1, depth):
            blocks.append(UNetBlock(in_channels=out_channels, out_channels=out_channels, emb_channels=embed_dim, use_hypnns=use_hypnns, version=version, ffwrd=ffwrd, lightweight=lightweight))

        return blocks

    def forward(self, x, site_id):

        latent_vector = self.embedding(site_id)

        x = self.pool(self.relu(self.norm1(self.conv1(x))))
        # latent_vector = latent_vector.repeat(x.shape[0]).view(x.shape[0], -1)
        for block in self.layer0:
            x = block(x, latent_vector)
        for block in self.layer1:
            x = block(x, latent_vector)
        for block in self.layer2:
            x = block(x, latent_vector)
        for block in self.layer3:
            x = block(x, latent_vector)

        out = self.fc(self.avgpool(x).reshape(-1, 512))
        return out
    
class UNetWithEmbedding(nn.Module):
    def __init__(self, num_classes, in_channels, embed_dim=2, encoder_layers=[2, 2, 2, 2], decoder_layers=[2, 2, 2, 2], site_number=1):
        super().__init__()

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for level, depth in enumerate(encoder_layers):
            self.enc.append(self._make_layer(
                depth=depth,
                in_channels=in_channels if level == 0 else 32 * 2**level,
                out_channels=64 * 2**level,
                embed_dim=embed_dim,
                down=True if level == len(encoder_layers) - 1 else False
            ))

        for level, depth in enumerate(decoder_layers):
            self.dec.append(self._make_layer(
                depth=depth,
            ))

    def _make_layer(self, depth, in_channels, out_channels, embed_dim, up=False, down=False):
        blocks = nn.ModuleList()

        for ndx in range(depth):
            blocks.append(UNetBlock(
                in_channels=in_channels if ndx == 0 else out_channels,
                out_channels=out_channels,
                emb_channels=embed_dim,
                up=up if ndx == depth - 1 else False,
                down=down if ndx == depth - 1 else False 
            ))

        return blocks

    def forward(self, x, site_id):
        return x