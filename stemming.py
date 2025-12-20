import nltk
from nltk.stem import SnowballStemmer

def setup_nltk_resources():
    """
    Controlla se le risorse necessarie di NLTK sono scaricate.
    Se mancano, le scarica automaticamente.
    """
    # 'punkt_tab' è richiesto nelle versioni recenti di NLTK per word_tokenize
    resources = ['punkt', 'punkt_tab'] 
    
    print("Verifica risorse NLTK...")
    for res in resources:
        try:
            # Cerca la risorsa nel sistema
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            print(f"Risorsa '{res}' non trovata. Download in corso...")
            try:
                nltk.download(res, quiet=True)
                print(f"'{res}' scaricata con successo.")
            except Exception as e:
                print(f"Errore durante il download di {res}: {e}")

# 1. Eseguiamo il setup una volta sola all'importazione del modulo
setup_nltk_resources()

# 2. Inizializziamo lo stemmer globale
stemmer = SnowballStemmer("english")

# 3. La funzione Tokenizer da passare al Vectorizer
def stemmed_tokenizer(text):
    # a. Tokenizza (spezza la frase in parole)
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        # Fallback di sicurezza se punkt fallisce ancora
        tokens = text.split()
    
    # b. Filtra caratteri non alfabetici e applica lo stemmer
    # isalpha() rimuove numeri e punteggiatura
    stems = [stemmer.stem(t) for t in tokens if t.isalpha()]
    
    return stems