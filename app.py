
# Esto es para deshabilitar la verificacion de certificados y asi evitar algun tipo de error a la hora de descargar los modelos
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Importacion de las librerias que se van a utilizar
import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
from gtts import gTTS

# Nombre: Dionis Metivier Santana
# Matricula: 24-EISN-2-022
# Proyecto Final: Calculador de Calorias

# Lista de Comidas (usando el dataset Food-101 descargado desde VsCode)
COMIDAS = [
    "apple pie", "baby back ribs", "baklava", "beef carpaccio", "beef tartare",
    "beet salad", "beignets", "bibimbap", "bread pudding", "breakfast burrito",
    "bruschetta", "caesar salad", "cannoli", "caprese salad", "carrot cake",
    "ceviche", "cheesecake", "cheese plate", "chicken curry", "chicken quesadilla",
    "chicken wings", "chocolate cake", "chocolate mousse", "churros", "clam chowder",
    "club sandwich", "crab cakes", "creme brulee", "croque madame", "cup cakes",
    "deviled eggs", "donuts", "dumplings", "edamame", "eggs benedict",
    "falafel", "filet mignon", "fish and chips", "french fries", "french toast",
    "fried calamari", "fried rice", "frozen yogurt", "garlic bread", "gnocchi",
    "greek salad", "grilled cheese sandwich", "grilled salmon", "guacamole",
    "gyoza", "hamburger", "hot dog", "hummus", "ice cream", "lasagna",
    "macaroni and cheese", "macarons", "miso soup", "mussels", "nachos",
    "omelette", "onion rings", "oysters", "pad thai", "paella", "pancakes",
    "panna cotta", "peking duck", "pho", "pizza", "pork chop", "poutine",
    "prime rib", "pulled pork sandwich", "ramen", "red velvet cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed salad", "shrimp and grits",
    "spaghetti bolognese", "spaghetti carbonara", "spring rolls", "steak",
    "strawberry shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna tartare", "waffles", "sandwich", "soup", "salad", "pasta", "rice",
    "fish", "chicken", "beef", "pork", "shrimp", "bread", "cake", "cookie",
    "pie", "coffee", "juice", "smoothie",

    # Frutas
    "apple", "banana", "orange", "strawberry", "watermelon", "grapes",
    "mango", "pineapple", "peach", "pear", "lemon", "lime", "cherry",
    "blueberry", "raspberry", "kiwi", "papaya", "coconut", "avocado",

    # Vegetales
    "tomato", "carrot", "broccoli", "spinach", "lettuce", "onion",
    "garlic", "potato", "sweet potato", "cucumber", "zucchini", "pepper",
    "corn", "mushroom", "eggplant", "cauliflower", "celery", "asparagus",
    "green beans", "peas", "cabbage", "beet", "radish", "artichoke"
]

# Base de datos para calorias local (por cada 100g), se utilizo la fuente USDA para estos datos
CALORIAS_DB = {
    "apple pie": 237, "baby back ribs": 290, "baklava": 428,
    "beef carpaccio": 150, "beef tartare": 174, "beet salad": 43,
    "beignets": 364, "bibimbap": 130, "bread pudding": 153,
    "breakfast burrito": 219, "bruschetta": 195, "caesar salad": 90,
    "cannoli": 327, "caprese salad": 120, "carrot cake": 415,
    "ceviche": 80, "cheesecake": 321, "cheese plate": 350,
    "chicken curry": 150, "chicken quesadilla": 219,
    "chicken wings": 290, "chocolate cake": 371,
    "chocolate mousse": 180, "churros": 357, "clam chowder": 75,
    "club sandwich": 290, "crab cakes": 172, "creme brulee": 217,
    "croque madame": 280, "cup cakes": 305, "deviled eggs": 145,
    "donuts": 452, "dumplings": 189, "edamame": 121,
    "eggs benedict": 247, "falafel": 333, "filet mignon": 267,
    "fish and chips": 290, "french fries": 312, "french toast": 229,
    "fried calamari": 175, "fried rice": 163, "frozen yogurt": 127,
    "garlic bread": 350, "gnocchi": 130, "greek salad": 70,
    "grilled cheese sandwich": 380, "grilled salmon": 208,
    "guacamole": 150, "gyoza": 210, "hamburger": 295,
    "hot dog": 290, "hummus": 166, "ice cream": 207,
    "lasagna": 135, "macaroni and cheese": 164, "macarons": 404,
    "miso soup": 40, "mussels": 86, "nachos": 306,
    "omelette": 154, "onion rings": 411, "oysters": 68,
    "pad thai": 196, "paella": 170, "pancakes": 227,
    "panna cotta": 130, "peking duck": 338, "pho": 55,
    "pizza": 266, "pork chop": 231, "poutine": 260,
    "prime rib": 340, "pulled pork sandwich": 280, "ramen": 436,
    "red velvet cake": 367, "risotto": 166, "samosa": 262,
    "sashimi": 130, "scallops": 88, "seaweed salad": 70,
    "shrimp and grits": 180, "spaghetti bolognese": 163,
    "spaghetti carbonara": 191, "spring rolls": 165,
    "steak": 271, "strawberry shortcake": 280, "sushi": 143,
    "tacos": 216, "takoyaki": 200, "tiramisu": 240,
    "tuna tartare": 120, "waffles": 291, "sandwich": 250,
    "soup": 60, "salad": 50, "pasta": 158, "rice": 130,
    "fish": 136, "chicken": 239, "beef": 250, "pork": 242,
    "shrimp": 99, "bread": 265, "cake": 350, "cookie": 480,
    "pie": 260, "coffee": 2, "juice": 45, "smoothie": 80,

    # Frutas
    "apple": 52, "banana": 89, "orange": 47, "strawberry": 32,
    "watermelon": 30, "grapes": 67, "mango": 60, "pineapple": 50,
    "peach": 39, "pear": 57, "lemon": 29, "lime": 30,
    "cherry": 50, "blueberry": 57, "raspberry": 52, "kiwi": 61,
    "papaya": 43, "coconut": 354, "avocado": 160,

    # Vegetales
    "tomato": 18, "carrot": 41, "broccoli": 34, "spinach": 23,
    "lettuce": 15, "onion": 40, "garlic": 149, "potato": 77,
    "sweet potato": 86, "cucumber": 16, "zucchini": 17,
    "pepper": 31, "corn": 86, "mushroom": 22, "eggplant": 25,
    "cauliflower": 25, "celery": 16, "asparagus": 20,
    "green beans": 31, "peas": 81, "cabbage": 25,
    "beet": 43, "radish": 16, "artichoke": 47
}

# Modelo de nutricion
class ModeloNutricion(nn.Module):
    def __init__(self):
        super(ModeloNutricion, self).__init__()
        self.fc = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.fc.weight, 1.0)

    def forward(self, x):
        return self.fc(x)


# 2. Cargar CLIP (la primera descargara 600MB, pero ya luego se quedara en el cache)
print("Cargando modelo CLIP...")
dispositivo  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_modelo  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dispositivo)
clip_proceso = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_modelo.eval()
print("CLIP listo.")

# Aranque del modelo de nutricion (incluye la evaluacion y desactivacion de gradientes y asi ahorrar la mayor cantidad de memoria posible)
modelo_nutricion = ModeloNutricion()
modelo_nutricion.eval()
for param in modelo_nutricion.parameters():
    param.requires_grad = False

# Funciones de la aplicacion
def detectar_comida(img_pil):
    try:
        textos = [f"a photo of {comida}" for comida in COMIDAS]
        
        # Preparando imagen y textos para el modelo
        inputs = clip_proceso(
            text=textos,
            images=img_pil.convert('RGB'),
            return_tensors="pt",
            padding=True
        ).to(dispositivo)

        with torch.no_grad():
            outputs   = clip_modelo(**inputs)
            logits    = outputs.logits_per_image
            probs     = logits.softmax(dim=1)
            indice    = probs.argmax().item()
            confianza = probs[0][indice].item()
            nombre    = COMIDAS[indice]

        print(f"[CLIP] Detectado: {nombre} ({confianza*100:.1f}% confianza)")
        return nombre
    except Exception as e:
        print(f"[CLIP] Error: {e}")
        return "food"


def obtener_calorias(nombre):
    if nombre in CALORIAS_DB:
        kcal = CALORIAS_DB[nombre]
        print(f"[DB] {nombre}: {kcal} kcal/100g")
        return float(kcal)

    for clave, kcal in CALORIAS_DB.items():
        if clave in nombre or nombre in clave:
            print(f"[DB] {nombre} → {clave}: {kcal} kcal/100g")
            return float(kcal)

    print(f"[DB] '{nombre}' no encontrado, usando valor por defecto")
    return 150.0


def analisis_plato(img_pil, gramos):
    if img_pil is None:
        return "Captura Imagen", None

    # Detectar la comida que se subio o se puso por la Webcam
    nombre_alimento = detectar_comida(img_pil)

    # Obtener calorias por 100g segun lo que tenemos en la base de datos local
    calorias        = obtener_calorias(nombre_alimento)
    
    # Calcular el total de calorias segun la cantidad indicada
    with torch.no_grad():
        base       = torch.tensor([[calorias / 100.0]])
        prediccion = modelo_nutricion(base)
        total      = round(prediccion.item() * float(gramos), 2)

    # Preparacion del mensaje que dio en el resultado
    Voz_resultado = (
        f"He identificado {nombre_alimento}. "
        f"Son aproximadamente {total} calorias en {gramos} gramos."
    )
    
    # Generador de audios en español usando gTTS
    tts = gTTS(text=Voz_resultado, lang='es')
    tts.save("resultado.mp3")
    
    # Regresar texto y ruta del audio a gradio
    return Voz_resultado, "resultado.mp3"


# Interfaz grafica con Gradio con diseño personalizado
css = """
    /* Fondo general de la aplicacion */
    .gradio-container {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460) !important;
        font-family: 'Arial', sans-serif;
    }

    /* Titulo principal */
    h1 {
        color: #e94560 !important;
        text-align: center !important;
        font-size: 2.5em !important;
        text-shadow: 0 0 20px #e94560 !important;
        margin-bottom: 5px !important;
    }

    /* Subtitulo */
    h3, p {
        color: #a8dadc !important;
        text-align: center !important;
    }

    /* Contenedor de columnas */
    .block {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(233, 69, 96, 0.3) !important;
        padding: 15px !important;
    }

    /* Boton principal */
    .primary {
        background: linear-gradient(90deg, #e94560, #0f3460) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-size: 1.1em !important;
        font-weight: bold !important;
        padding: 12px !important;
        box-shadow: 0 0 15px rgba(233, 69, 96, 0.5) !important;
    }

    .primary:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 25px rgba(233, 69, 96, 0.8) !important;
    }

    /* Labels de los componentes */
    label {
        color: #a8dadc !important;
        font-weight: bold !important;
    }

    /* Cuadro de texto de resultado */
    textarea {
        background: rgba(15, 52, 96, 0.6) !important;
        color: #ffffff !important;
        border: 1px solid #a8dadc !important;
        border-radius: 10px !important;
        font-size: 1.1em !important;
    }

    /* Componente de imagen */
    .image-container {
        border-radius: 15px !important;
        border: 2px solid #e94560 !important;
        overflow: hidden !important;
    }

    /* Reproductor de audio */
    .audio-container {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        border: 1px solid #a8dadc !important;
    }

    /* Campo numerico */
    input[type=number] {
        background: rgba(15, 52, 96, 0.6) !important;
        color: white !important;
        border: 1px solid #a8dadc !important;
        border-radius: 8px !important;

     }
"""

with gr.Blocks(css=css) as interfaz:
    gr.Markdown("Calculador de Calorias")
    gr.Markdown("Sube una imagen o utilizar la Webcam para analizar la comida")

    with gr.Row():
        
        with gr.Column():
            img_input  = gr.Image(sources=["webcam", "upload"], type="pil", label="Camara en Vivo")
            peso_input = gr.Number(label="Gramos de la porcion", value=100)
            btn        = gr.Button("Analisis de Plato", variant="primary")

        with gr.Column():
            txt_output = gr.Textbox(label="Resultado del Analisis")
            aud_output = gr.Audio(label="Audio", autoplay=True)

    btn.click(analisis_plato, [img_input, peso_input], [txt_output, aud_output])

#Arranque principal de la aplicacion completa (le puse el share=True para que funcionara tanto de manera local y publico)
if __name__ == "__main__":
    interfaz.launch(theme='Soft', share=True)