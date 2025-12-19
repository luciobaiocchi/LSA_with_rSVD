# LSA_with_rSVD
Latent semantic analysis with randomized singular value decomposition

Il dataset è una raccolta di circa 18.000 post presi da newsgroup (i vecchi forum di discussione Usenet) degli anni '90. I post sono suddivisi in 20 categorie (topic) diverse. Alcune categorie sono molto diverse tra loro (es. sci.space vs rec.sport.hockey), mentre altre sono molto simili (es. comp.sys.ibm.pc.hardware vs comp.sys.mac.hardware).

Viene usato principalmente per:

- Classificazione del testo: Addestrare algoritmi per capire automaticamente l'argomento di un testo (es. "Questo testo parla di baseball o di medicina?").

- Clustering: Raggruppare documenti simili senza conoscere le etichette a priori.

- Benchmarking: Testare la velocità e la precisione di nuovi algoritmi di Machine Learning su dati testuali reali (ma puliti).

### Rimuoviamo dati inutili
remove: Molto importante. Accetta una tupla come ('headers', 'footers', 'quotes'). Serve a rimuovere le intestazioni delle email, le firme e le citazioni.