# convtt
Convolutional Neural Netoworks Training Tools

## example of DenseNet training

The example is included in the `bin` folder in the package with the file name of `convtt_train_densenet.py`. 

Here is a code snippet of creating a trainer and train the network given the model and the dataset. 

```python
# convtt_train_densenet.py
from convtt.models import densenet
from convtt.train.trainer import *

# initialise trainer
optimiser = build_optimiser(model=model, name='ScheduledSGD', milestones=[10, 20], lr=0.1)
driver = build_driver(model=model, training_epoch=30, batch_size=128, training_data=dataset.train['images'],
                      training_label=dataset.train['labels'],
                      validation_data=None, validation_label=None, test_data=dataset.test['images'],
                      test_label=dataset.test['labels'], optimiser=optimiser)
trainer = build_trainer(optimiser=optimiser, driver=driver)
test_acc = trainer.eval()
print(test_acc)
```
