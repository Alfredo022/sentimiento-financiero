# Preprocesamiento de texto financiero (usado por el notebook y por la app de Streamlit)
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

for _r in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords",
           "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(_r, quiet=True)

STOP = set(stopwords.words("english"))
# Palabras que la lista estándar elimina pero que cambian el sentido de una frase financiera:
# negaciones (hallazgo de la Tarea 2) y palabras de dirección ("up from", "down to")
CONSERVAR = {"no", "nor", "not", "up", "down", "above", "below", "over", "under", "off",
             "against", "more", "most"}
STOP_MEJORADA = STOP - CONSERVAR
_lemm = WordNetLemmatizer()

def _pos_wordnet(etiqueta):
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(etiqueta[0], wordnet.NOUN)

def limpiar_texto(texto):
    """Limpieza común: caracteres corruptos, minúsculas, emojis, menciones/enlaces y números."""
    t = emoji.demojize(texto, delimiters=(" ", " ")).replace("_", " ")   # emojis -> palabras (antes de quitar no-ASCII)
    t = re.sub(r"\+?[^\x00-\x7F]", "", t)               # artefactos de codificación: "Sepp+ñl+ñ" -> "Seppl"
    t = t.lower()
    t = re.sub(r"@\w+|#\w+|http\S+|www\S+", " ", t)              # menciones, hashtags y enlaces
    t = re.sub(r"\d+(?:[.,]\d+)*", " num ", t)                      # cifras -> marcador "num"
    return t

def preprocesar_mejorado(texto):
    """Pipeline final: conserva negaciones y palabras de dirección, orden y repeticiones; lematiza con POS."""
    tokens = word_tokenize(limpiar_texto(texto))
    tokens = ["not" if w == "n't" else w for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in STOP_MEJORADA]
    return " ".join(_lemm.lemmatize(w, _pos_wordnet(tg)) for w, tg in nltk.pos_tag(tokens))

def preprocesar_t2(texto):
    """Función de la Tarea 2 tal cual (stopwords completas, sin repetidos, orden alfabético)."""
    t = emoji.demojize(texto, delimiters=(" ", " ")).replace("_", " ")
    t = re.sub(r"\+?[^\x00-\x7F]", "", t).lower()
    t = re.sub(r"@\w+|#\w+|http\S+|www\S+|https\S+", "", t)
    tokens = [w for w in word_tokenize(t) if w.isalpha()]
    tokens = [w for w in tokens if w not in STOP]
    return " ".join(sorted(set(_lemm.lemmatize(w) for w in tokens)))
