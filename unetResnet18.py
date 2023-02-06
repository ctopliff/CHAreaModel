import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml

from _resnets import resnet18

class UNetResnet18(pl.LightningModule):
    def __init__(self, ckpt_path=None, learning_rate=1e-4, batch_size=32, reg = 1e-3, resize=512, crop=300, n_classes=2, n_channels=3, freeze=True, skip_connections=True, num_dconvs = 4, embedding_size=16, reg_embedding_size = 16, out_size=1):
        super(UNetResnet18, self).__init__()
        self.save_hyperparameters()
        if ckpt_path:
            with open(ckpt_path + 'config.yaml', 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            self.u_net = UNetScaling.load_from_checkpoint(ckpt_path + 'checkpoints/best_model.ckpt', resize=config['data']['resize'], 
                                                          n_channels=config['model']['n_channels'], n_classes=config['model']['n_classes'],
                                                          num_dconvs=config['model']['num_dconvs'], embedding_size=config['model']['embedding_size'], 
                                                          skip_connections=config['model']['skip_connections'])
            embedding_size = config['model']['embedding_size']
        else:
            self.u_net = UNetScaling(resize=resize, n_channels=n_channels, n_classes=n_classes, skip_connections=skip_connections, num_dconvs=num_dconvs, embedding_size=embedding_size)

        self.u_net.decoder_layers[-1] = nn.Identity()
        if freeze:
            self.u_net.freeze()

        self.resnet18 = resnet18()
        self.resnet18.conv1 = nn.Conv2d(embedding_size, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=1000, bias=True)
        self.outLayer = nn.Sequential(nn.Dropout(.5),
                                      nn.ReLU(inplace=True),
                                      #nn.Linear(in_features=1000, out_features=1, bias=True) #for a single target
                                      nn.Linear(in_features=1000, out_features=out_size, bias=True) # for two targets (maybe this should auto scale?)
        )

    def forward(self, x):

        logits, embedding = self.u_net(x)
        # logits = logits.flatten(start_dim=1)
        out = self.resnet18(logits)
        out = self.outLayer(out).squeeze()

        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        out = self.forward(x)
        loss = F.mse_loss(out, y)

        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        out = self.forward(x)
        loss = F.mse_loss(out, y)

        self.log("hp/val_loss", loss, on_epoch=True, on_step=False)
        return {'pred':out, 'labl': y}

    def validation_epoch_end(self, val_step_outputs):
        predictions = []
        labels = []

        for item in val_step_outputs:
            predictions.append(item['pred'])
            labels.append(item['labl'])

        val_rmse = torch.sqrt(torch.mean((torch.stack(labels) - torch.stack(predictions))**2))

        self.log("hp/val_rmse", val_rmse, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):

        x, y = batch
        _, embedding = self.u_net(x)
        out = self.forward(x)
        loss = F.mse_loss(out, y.squeeze())

        return {'pred':out.cpu(), 'emb':embedding.cpu(), 'labl':y.squeeze().cpu()}

    def test_epoch_end(self, test_step_outputs):
        predictions = []
        embeddings = []
        labels = []

        for item in test_step_outputs:
            predictions.append(item['pred'])
            embeddings.append(item['emb'])
            labels.append(item['labl'])
        test_rmse = torch.sqrt(torch.mean((torch.stack(labels) - torch.stack(predictions))**2))

        self.log("hp/final_rmse", test_rmse, on_epoch=True, on_step=False)

        self.test_predictions = torch.stack(predictions)
        self.test_embeddings = torch.cat(embeddings)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)

        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=.5),
            'monitor': 'hp/val_loss'
        }