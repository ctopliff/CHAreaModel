""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import pytorch_lightning as pl

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = out.permute(0, 2, 3, 1).contiguous().view(-1, self.out_channels)
        y = y.view(-1)
        
        loss = F.cross_entropy(out, y.long())
        self.log("train_loss", loss, on_epoch=True,
                 on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        out = self.forward(x)
        
        out = out.permute(0, 2, 3, 1).contiguous().view(-1, self.out_channels)
        y = y.view(-1)
        
        loss = F.cross_entropy(out, y.long())

        self.log("hp/val_loss", loss, sync_dist=True, on_epoch=True, on_step=False)

        return loss
    
    def test_step(self, batch, batch_idx):

        x, y = batch
        out = self.forward(x)
        
        out = out.permute(0, 2, 3, 1).contiguous().view(-1, self.out_channels)
        y = y.view(-1)
        
        loss = F.cross_entropy(out, y.long())

        return {'pred':out.cpu(), 'labl':y.squeeze().cpu()}

    def test_epoch_end(self, test_step_outputs):
        predictions = []
        labels = []

        for item in test_step_outputs:
            predictions.append(item['pred'])
            labels.append(item['labl'])
        test_loss = F.cross_entropy(torch.stack(predictions).view(-1, self.out_channels, self.im_size, self.im_size), torch.stack(labels).view(-1, self.im_size, self.im_size).long())

        self.log("hp/final_loss", test_loss, on_epoch=True, on_step=False)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return {
            'optimizer':optimizer, 
            'lr_scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=.5),
            'monitor': 'hp/val_loss'
        }