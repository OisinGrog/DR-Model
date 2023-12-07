import pytorch_lightning as pl
import torch.optim
from test_pretrained import pretrained_Resnet
import torch.nn as nn
import torchmetrics


class DR_model(pl.LightningModule):
    def __init__(self, learning_rate):
        super(DR_model, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = float(learning_rate)
        self.model = pretrained_Resnet()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        inputs = x['image']
        output = self.model(inputs)
        return output

    def _evaluate(self, data):
        label = data['DR_label']
        # print(f'Shape of label: {label.shape}')
        class_output = self(data)
        # print(f'Class Output shape: {class_output.shape}')
        loss = self.cross_entropy(class_output, label)
        acc = self.accuracy(class_output.argmax(dim=1), label)
        return {'loss': loss, 'accuracy':acc}

    def _step(self, batch, step_name):
        res = self._evaluate(batch)
        self.log_dict({f'{step_name}/{key}': val for key, val in res.items()}, prog_bar=True, on_epoch=True,
                      logger=True, sync_dist=True)
        return res

    def training_step(self, batch, _):
        return self._step(batch, 'train')

    def test_step(self, batch, _):
        return self._step(batch, 'test')

    def validation_step(self, batch, _):
        return self._step(batch, 'valid')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

