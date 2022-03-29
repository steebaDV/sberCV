import torch
from torchvision import transforms, models
from torch.nn.functional import softmax

import numpy as np

DICT_LABELS = {
    155: 0,
    159: 1,
    162: 2,
    167: 3,
    182: 4,
    193: 5,
    207: 6,
    229: 7,
    258: 8,
    273: 9,
}

DICT_BREEDS = {
    0: 'Shih-Tzu',
    1: 'Rhodesian ridgeback',
    2: 'Beagle',
    3: 'English foxhound',
    4: 'Border terrier',
    5: 'Australian terrier',
    6: 'Golden retriever',
    7: 'Old English sheepdog',
    8: 'Samoyed',
    9: 'Dingo',
}


class Model(torch.nn.Module):
    def __init__(self, breed_labels=None):
        super(Model, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.breed_labels = breed_labels

    def forward(self, x):
        x = self.resnet(x)
        if self.breed_labels is not None:
            x = x[:, self.breed_labels]
        x = softmax(x, dim=1)
        return x


class Pipeline:
    def __init__(self, Model, transform, device=None):
        self.dict_labels = DICT_LABELS
        self.dict_breeds = DICT_BREEDS
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transform
        self.Model = Model
        self.model = self.load_model()

    def get_breeds_list(self):
        breeds_list = [self.dict_breeds[i] for i in range(len(self.dict_labels))]
        return breeds_list

    def load_model(self):
        breed_labels = sorted(list(self.dict_labels.keys()))

        model = self.Model(breed_labels=breed_labels)
        model.eval()
        model.to(self.device)
        return model

    def preprocess(self, img):
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    def predict_proba(self, img):
        tensor = self.preprocess(img)
        probas = self.model(tensor).data.cpu().numpy()
        return probas

    def predict(self, img):
        labels = self.predict_proba(img).argmax(axis=1)
        breeds = np.array(list(map(self.dict_breeds.get, labels)))
        return breeds


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])



