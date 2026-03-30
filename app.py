import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_model
from datasets import load_dataset
from tqdm import tqdm
import gradio as gr
from gtts import gTTS
import io
import requests

# Mis Datos:
# Dionis Metivier Santana
# 24-EISN-2-022
# Proyecto Final: Calculador de Calorias de Comidas

EDAMAM_ID = "c1b7ea6f"
EDAMAM_KEY = "436e2d11559c0e38c012687272750201"
MODELO = "food101_model.safetensors"

FOOD101 = [
     'apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare',
    'beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito',
    'bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake',
    'ceviche','cheesecake','cheese_plate','chicken_curry','chicken_quesadilla',
    'chicken_wings','chocolate_cake','chocolate_mousse','churros','clam_chowder',
    'club_sandwich','crab_cakes','creme_brulee','croque_madame','cup_cakes',
    'deviled_eggs','donuts','dumplings','edamame','eggs_benedict','escargots',
    'falafel','filet_mignon','fish_and_chips','foie_gras','french_fries',
    'french_onion_soup','french_toast','fried_calamari','fried_rice','frozen_yogurt',
    'garlic_bread','gnocchi','greek_salad','grilled_cheese_sandwich','grilled_salmon',
    'guacamole','gyoza','hamburger','hot_and_sour_soup','hot_dog','huevos_rancheros',
    'hummus','ice_cream','lasagna','lobster_bisque','lobster_roll_sandwich',
    'macaroni_and_cheese','macarons','miso_soup','mussels','nachos','omelette',
    'onion_rings','oysters','pad_thai','paella','pancakes','panna_cotta',
    'peking_duck','pho','pizza','pork_chop','poutine','prime_rib','pulled_pork_sandwich',
    'ramen','red_velvet_cake','risotto','samosa','sashimi','scallops','seaweed_salad',
    'shrimp_and_grits','spaghetti_bolognese','spaghetti_carbonara','spring_rolls',
    'steak','strawberry_shortcake','sushi','tacos','takoyaki','tiramisu',
    'tuna_tartare','waffles'
]

class Food101Dataset(Dataset):
     def __init__(self, dataset, transform=None):
          self.dataset = dataset
          self.transform = transform

     def __len__(self):
           return len(self.dataset)
     
     def __getitem__(self, idx):
          image = self.dataset [idx]['image'].convert('RGB')
          label = self.dataset[idx]['label']
          if self.transform:
               image = self.transform(image)
          return image, label

#1. Cerebro de la aplicacion para recibir datos de las comidas y saber como procesarlas para darnos el resultado final
class ModeloComida (nn.Module):
    def __init__(self, num_clases=101):
        super(ModeloComida, self).__init__()
        self.base = models.efficientnet_b0(
             weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.base.classifier[1] = nn.Linear(
             self.base.classifier[1].in_features, num_clases
        )
    def forward(self, x):
        return self.base(x)
    
class ModeloNutricion(nn.Module):
         def __init__(self):
              super(ModeloNutricion, self).__init__()
              self.fc = nn.Linear(1, 1, bias=False)
              nn.init.constant_(self.fc.weight, 1.0)
         
         def forward(self, x):
              return self.fc(x)

if __name__ == "__main__":
    interfaz.launch(theme='Soft')