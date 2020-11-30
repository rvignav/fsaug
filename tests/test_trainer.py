from fsaug.net import Net
from fsaug.trainer import Trainer

network = Net()
trainer = Trainer(network)
trainer.log_details()
trainer.run()