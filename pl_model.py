import pytorch_lightning as pl
import torch.optim
from models import pretrained_Resnet
import torch.nn as nn
import torchmetrics


total_instances = 15220 + 1046
class_0_weight = 1/(15220 * total_instances)
class_1_weight = 1/(1046 * total_instances)

# Normalize the weights such that the smallest weight is 1
max_weight = max(class_0_weight, class_1_weight)
class_0_weight = class_0_weight / max_weight
class_1_weight = class_1_weight / max_weight

class_weights = torch.tensor([class_0_weight, class_1_weight], dtype=torch.float32)
print(class_weights)

class DR_model(pl.LightningModule):
    def __init__(self, learning_rate):
        super(DR_model, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = float(learning_rate)
        self.model = pretrained_Resnet()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
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
        return {'loss': loss, 'acc': acc}

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'valid/acc'
            }
