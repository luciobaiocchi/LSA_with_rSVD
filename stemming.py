import nltk
from nltk.stem import SnowballStemmer

def setup_nltk_resources():
    """
    Controlla se le risorse necessarie di NLTK sono scaricate.
    Se mancano, le scarica automaticamente.
    """
    resources = ['punkt', 'punkt_tab'] # 'punkt_tab' serve per le versioni nuove di NLTK
    
    for res in resources:
        try:
            # Cerca la risorsa nel sistema
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            print(f"Risorsa '{res}' non trovata. Download in corso...")
            nltk.download(res, quiet=True)
            print(f"'{res}' scaricata.")

# 1. Eseguiamo il setup una volta sola all'avvio
setup_nltk_resources()

# 2. Inizializziamo lo stemmer globale (così non lo ricrea per ogni parola)
stemmer = SnowballStemmer("english")

# 3. La funzione Tokenizer da passare al Vectorizer
def stemmed_tokenizer(text):
    # a. Tokenizza (spezza la frase in parole)
    # Gestisce punteggiatura e spazi meglio dello split() base
    tokens = nltk.word_tokenize(text)
    
    # b. Filtra caratteri non alfabetici e applica lo stemmer
    # isalpha() rimuove numeri e punteggiatura residua (es. "1993", "?")
    stems = [stemmer.stem(t) for t in tokens if t.isalpha()]
    
    return stems