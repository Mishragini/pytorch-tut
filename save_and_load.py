import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(),'model_weights.pth')

model = models.vgg16()  # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model.weights.pth',weight_only=True))
model.eval()

#with shapes
torch.save(model,'model.pth')

#legacy use
model = torch.load('model.pth',weights_only=False)