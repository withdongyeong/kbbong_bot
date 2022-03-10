import torch
import torchvision.models as models

squeezenet = models.squeezenet1_0()
torch.save(squeezenet, "aa.pickle")

test = torch.load("aa.pickle")

print(test)