# AutoRC – Modulo di Estrazione Dataset da Rosbag2
Il progetto costituisce il lavoro svolto nell’ambito della mia **tesi triennale in Ingegneria Informatica presso l’Università degli Studi di Firenze**, dedicata alla raccolta dati e creazione di un dataset di traiettorie per l’addestramento di modelli di guida autonoma.

Questo repository contiene il modulo software sviluppato per l’estrazione di campioni (sample) da registrazioni **rosbag2** ottenute dal modello AutoRC, un veicolo radiocomandato equipaggiato con sensori per la guida autonoma.  

## Scopo del Progetto

Il codice fornisce uno strumento completo per:

- Elaborare dati raccolti con il modello AutoRC  
- Generare dataset di traiettorie per il training di reti neurali e modelli di predizione del movimento  
- Allineare coerentemente dati 2D (RGB, depth) e 3D (point-cloud e traiettorie)  
- Supportare studi e sviluppi nel campo della guida autonoma

---

## Struttura del Modulo

Il software è organizzato attorno a tre classi principali.

### Sequence
Rappresenta una singola sequenza di registrazione contenuta in una cartella rosbag2.

Responsabilità principali:

- lettura dei messaggi ROS2 dai topic:
  - `/zed/zed_node/odom` (odometria)
  - `/zed/zed_node/right/image_rect_color` (frame RGB)
  - `/zed/zed_node/depth/depth_registered` (mappe di profondità)
- gestione dei timestamp e del numero di frame della sequenza
- accesso alla matrice intrinseca della fotocamera ZED
- generazione dei campioni tramite `get_sample()`

### Sample
Rappresenta un singolo campione estratto da una sequenza contenente:

- nuvola di punti (point-cloud) ricostruita dalla fotocamera stereoscopica ZED 2i  
- frame RGB  
- frame di profondità  
- traiettoria passata  
- traiettoria futura

Include inoltre metodi per la visualizzazione e l’analisi del singolo campione.

### Trajectory
Modella una sequenza di punti tridimensionali, utilizzata per rappresentare la traiettoria passata e quella futura di ciascun campione.


## Funzione Principale: `get_sample()`
Il metodo `get_sample(t, delta_f, delta_p, framerate, max_velocity)` è la componente centrale del modulo e consente di generare un campione a partire dal frame indice *t* all’interno della sequenza.

Il metodo svolge le seguenti operazioni:

- costruzione della traiettoria futura entro un intervallo di `delta_f` secondi
- costruzione della traiettoria passata entro un intervallo di `delta_p` secondi
- interpolazione lineare della posizione nei timestamp non allineati
- interpolazione quadratica per garantire un numero di punti coerente con il framerate richiesto
- verifica della validità del campione in base a:
  - disponibilità dei dati per coprire gli intervalli richiesti
  - numero minimo di punti per poter interpolare
  - limite massimo di velocità (`max_velocity`) per eliminare campioni incoerenti

Durante la generazione, il metodo associa al frame odometrico il frame RGB e depth più vicini temporalmente e genera la nuvola di punti tramite proiezione 3D.  
Point-cloud e traiettorie vengono poi portate nello stesso sistema di riferimento tramite rototraslazioni basate sull’orientamento della fotocamera ZED.

---

## Generazione del Dataset

La procedura di generazione del dataset è automatizzata dallo script dedicato `dataset_maker.py`, che si occupa di caricare le cartelle rosbag2, creare gli oggetti Sequence, iterare sui frame generando i campioni tramite `get_sample()` e salvare ogni campione valido come file `.npz` individuale nella directory di output.  
Questo formato permette di caricare rapidamente i singoli campioni o insiemi di essi durante la fase di addestramento delle reti neurali.

---

## Visualizzazione

Il modulo fornisce metodi di visualizzazione basati su Open3D.

### `display()`
Mostra in un’unica finestra:

- nuvola di punti  
- traiettoria passata  
- traiettoria futura  

tutte allineate nel medesimo sistema di riferimento tridimensionale.

### `display_sequence()`
Ricostruisce l’intera scena della sequenza creando un mosaico di point-cloud, ciascuna posizionata secondo l’odometria corrispondente al frame di acquisizione.  
Il sistema di riferimento è fissato sulla nuvola di punti del primo frame della sequenza.

---

## Input Richiesto

Il modulo richiede cartelle rosbag2 contenenti almeno i seguenti topic:

- `/zed/zed_node/odom`
- `/zed/zed_node/right/image_rect_color`
- `/zed/zed_node/depth/depth_registered`

Ogni cartella deve includere:

- file `.db3` con i messaggi registrati  
- file `metadata.yaml` con la descrizione della registrazione

---

## Requisiti

- Python 3.x  
- numpy  
- Open3D  
- cv2  
- rosbag2_py 




