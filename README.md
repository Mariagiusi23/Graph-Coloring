# 🎨 Colorazione di grafi con GNN ed energia di Potts differenziabile

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-GNN-3C2179)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Algorithms-orange)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Descrizione

Questa repository contiene l’implementazione di una pipeline per la **colorazione di grafi** basata su **Graph Neural Network** e su una rilassazione differenziabile dell’**energia antiferromagnetica di Potts**.

L’obiettivo è assegnare un colore a ciascun nodo di un grafo non orientato, usando un numero fissato di colori `c`, in modo che due nodi adiacenti non abbiano lo stesso colore.

La pipeline combina:

- 📂 parsing di grafi in formato DIMACS;
- 🔗 conversione in strutture dati compatibili con PyTorch Geometric;
- 🧠 embedding trainabili dei nodi;
- 🕸️ message passing con connessioni residuali;
- ⚡ loss differenziabile ispirata al modello di Potts;
- 🎯 discretizzazione tramite `argmax`;
- 🛠️ post-processing con euristica locale `min-conflicts`;
- 📊 confronto con baseline greedy di NetworkX;
- 🌈 visualizzazione dei risultati.

Gli esperimenti sono stati condotti sulle istanze DIMACS `g6` e `g12`.

## 🎓 Contesto del progetto

Progetto sviluppato per il corso:

**Applicazioni Fisiche al Machine Learning**  
**Prof. Stefano Giagu**
**Sapienza Università di Roma**  
**A.A. 2025–2026**

Autrici:

- **Eleonora De Cicco**
- **Mariagiusi Nicodemo**

## 🧩 Formulazione del problema

Dato un grafo non orientato:

```math
G = (V, E)
```

una colorazione con `c` colori assegna a ogni nodo un colore:

```math
q_i \in \{1, \dots, c\}
```

La colorazione è valida se, per ogni arco `(i, j)`, i due estremi hanno colori diversi:

```math
q_i \neq q_j
```

Il numero di conflitti è definito come:

```math
N_\mathrm{conf}(q) =
\sum_{(i,j)\in E} \mathbf{1}[q_i = q_j]
```

Una colorazione valida richiede quindi:

```math
N_\mathrm{conf} = 0
```

Poiché la determinazione del numero cromatico è un problema combinatorio difficile, il problema viene riformulato come ottimizzazione differenziabile.

## 🧠 Metodo

La Graph Neural Network produce, per ogni nodo, una distribuzione continua sui colori:

```math
p_i = (p_{i1}, \dots, p_{ic})
```

La loss di Potts penalizza la sovrapposizione tra le distribuzioni di nodi adiacenti:

```math
\mathcal{L}_\mathrm{Potts}
=
\frac{1}{|E|}
\sum_{(i,j)\in E}
p_i^\top p_j
```

Alla loss viene aggiunto un termine entropico, con peso `λ = 0.02`, per favorire distribuzioni più concentrate:

```math
\mathcal{L}_\mathrm{tot}
=
\mathcal{L}_\mathrm{Potts}
+
\lambda H
```

Dopo l’addestramento, l’assegnazione continua viene trasformata in una colorazione discreta tramite:

```math
q_i = \arg\max_k p_{ik}
```

La colorazione ottenuta viene poi raffinata con una procedura di repair locale di tipo **min-conflicts**.

## 🏗️ Architettura

La rete utilizza:

- 🧬 embedding trainabile di dimensione `64` per ciascun nodo;
- 🔁 due layer di message passing;
- ➕ connessioni residuali;
- ⚙️ attivazioni ReLU;
- 🎨 layer lineare finale con `c` output;
- 📌 softmax finale sui colori.

Ogni layer di message passing aggiorna la rappresentazione di un nodo combinando la sua rappresentazione corrente con la media delle rappresentazioni dei nodi vicini.

## 📊 Risultati principali

La scansione robusta su cinque seed ha trovato colorazioni valide GNN+repair con:

| Grafo | Migliore greedy | Migliore GNN+repair | Success rate alla soglia |
|------|------------------|----------------------|---------------------------|
| `g6` | 13 colori | 11 colori | 40% |
| `g12` | 31 colori | 35 colori | 40% |

✅ La pipeline GNN+repair migliora la baseline greedy su `g6`.  
⚠️ Su `g12`, invece, la baseline greedy fornisce una colorazione migliore.

Per `g12`, la baseline greedy ottiene infatti una colorazione valida con **31 colori**, mentre la pipeline neurale trova una colorazione valida con **35 colori**.

I risultati mostrano inoltre che la sola GNN non raggiunge direttamente zero conflitti vicino alla soglia osservata: il post-processing locale è essenziale per ottenere colorazioni valide.

## 📁 Struttura della repository

```text
.
├── graph_coloring.ipynb
├── relazione_DeCicco_Nicodemo.pdf
├── README.md
├── LICENSE
├── requirements.txt
└── results
```

## ⚙️ Setup

### 1️⃣ Clonare la repository

```bash
git clone https://github.com/Mariagiusi23/Graph-Coloring.git
cd Graph-Coloring
```

### 2️⃣ Creare un ambiente virtuale

```bash
python -m venv .venv
```

Attivazione su Linux/macOS:

```bash
source .venv/bin/activate
```

Attivazione su Windows:

```bash
.venv\Scripts\activate
```

### 3️⃣ Installare le dipendenze

```bash
pip install -r requirements.txt
```

A seconda della versione di CUDA e del sistema operativo, `torch` e `torch-geometric` potrebbero richiedere installazioni specifiche dai rispettivi siti ufficiali.

## ▶️ Esecuzione del notebook

Per eseguire il progetto localmente:

```bash
jupyter notebook graph_coloring.ipynb
```

oppure:

```bash
jupyter lab
```

Il progetto può essere eseguito anche direttamente su Google Colab tramite il seguente link:

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VjJHC8o4940Sf6CJBONgPTW135KaMmF2#scrollTo=53ed4bcd)

Questo permette di riprodurre gli esperimenti senza configurare manualmente l’ambiente locale, sfruttando le risorse disponibili su Colab.
Il notebook contiene:

- 📥 caricamento e parsing dei grafi;
- 🧱 costruzione delle strutture dati;
- 🧠 definizione della GNN;
- 🏋️ training loop;
- ❌ calcolo dei conflitti;
- 🛠️ procedura di repair;
- 🔍 scansione sul numero di colori `c`;
- 📊 confronto con baseline greedy;
- 🌈 grafici e visualizzazioni finali.

## 🔁 Riproducibilità

Gli esperimenti sono stati eseguiti usando più seed casuali.

La scansione robusta utilizza cinque seed:

```python
seeds = [0, 1, 2, 3, 4]
```

Per ogni grafo e per ogni numero di colori `c`, la rete viene addestrata da zero.

Parametri principali:

```python
learning_rate = 1e-2
weight_decay = 1e-5
epochs = 700
hidden_dim = 64
entropy_lambda = 0.02
max_repair_steps = 3000
```

Per rendere gli esperimenti più riproducibili, è consigliato fissare i seed prima di ogni run:

```python
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

e chiamare:

```python
set_seed(seed)
```

prima dell’addestramento.

La riproducibilità esatta può comunque dipendere da:

- 💻 hardware utilizzato;
- 🐍 versione di Python;
- 🔥 versione di PyTorch;
- 🧩 versione di PyTorch Geometric;
- 🚀 eventuale uso di CUDA;
- 🎲 operazioni GPU non deterministiche.

Per una maggiore riproducibilità, si consiglia l’esecuzione su CPU oppure l’uso delle impostazioni deterministiche di PyTorch, quando disponibili.

## 🧪 Protocollo sperimentale

Per ogni istanza del grafo:

1. si sceglie un numero di colori `c`;
2. si inizializza la GNN da zero;
3. si addestra la rete minimizzando la loss di Potts differenziabile;
4. si monitora il numero di conflitti discreti durante il training;
5. si salva la migliore colorazione raw osservata;
6. si applica il repair locale min-conflicts;
7. si verifica la colorazione finale sugli archi unici;
8. si ripete l’esperimento su più seed;
9. si confronta il risultato con baseline greedy.

I valori finali devono essere interpretati come **upper bound sperimentali**, non come dimostrazioni di ottimalità del numero cromatico.

## 🏁 Baseline greedy

Come confronto vengono utilizzate strategie greedy di NetworkX, tra cui:

- `largest_first`;
- `smallest_last`;
- `saturation_largest_first`.

Il miglior risultato greedy viene usato come baseline di riferimento.

## 💡 Osservazioni principali

- Una loss continua bassa non garantisce automaticamente una colorazione discreta valida.
- La discretizzazione tramite `argmax` può introdurre conflitti.
- Il repair locale è determinante vicino alla soglia di colorabilità osservata.
- Il metodo è sensibile all’inizializzazione.
- La pipeline GNN+repair migliora il greedy su `g6`.
- Su `g12`, invece, la baseline greedy ottiene un risultato migliore.
- Un metodo neurale più complesso non è necessariamente superiore a euristiche combinatorie semplici.

## 🚀 Possibili sviluppi

Possibili miglioramenti futuri includono:

- aumento del numero di restart vicino alla soglia;
- early stopping basato sui conflitti discreti;
- scheduling del learning rate;
- annealing della temperatura softmax;
- variazione del peso del termine entropico;
- repair più forti, come tabu search o simulated annealing;
- inizializzazione greedy;
- studi di ablazione sul numero di layer di message passing;
- confronto tra repair inizializzato da GNN, greedy e assegnazioni casuali;
- misura sistematica di tempi, memoria e passi di repair;
- confronto con solver esatti o lower bound.

## 📚 Riferimenti bibliografici

- **[1] T. R. Jensen, B. Toft — _Graph Coloring Problems_**, Wiley-Interscience, 1995.  
  Libro di riferimento sul problema della colorazione di grafi e sulle sue principali formulazioni teoriche.  
  🔗 [Wiley Online Library](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118032497)  
  🔗 [Archivio degli autori](https://www.imada.sdu.dk/Research/Graphcol/)

- **[2] T. N. Kipf, M. Welling — _Semi-Supervised Classification with Graph Convolutional Networks_**, ICLR, 2017.  
  Articolo fondamentale sulle Graph Convolutional Networks, rilevante per il message passing su grafi.  
  🔗 [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

- **[3] W. L. Hamilton — _Graph Representation Learning_**, Morgan & Claypool, 2020.  
  Testo introduttivo e approfondito su embedding di grafi, Graph Neural Network e rappresentazioni neurali su grafi.  
  🔗 [Pagina del libro](https://www.cs.mcgill.ca/~wlh/grl_book/)  
  🔗 [PDF preprint](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)

- **[4] M. J. A. Schuetz, J. K. Brubaker, H. G. Katzgraber — _Graph coloring with physics-inspired graph neural networks_**, Physical Review Research, 4, 043131, 2022.  
  Lavoro direttamente collegato all’approccio physics-inspired per la colorazione di grafi tramite GNN e modello di Potts.  
  🔗 [arXiv:2202.01606](https://arxiv.org/abs/2202.01606)  
  🔗 [Physical Review Research](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.043131)

- **[5] S. Minton, M. D. Johnston, A. B. Philips, P. Laird — _Minimizing conflicts: a heuristic repair method for constraint satisfaction and scheduling problems_**, Artificial Intelligence, 58(1–3), 161–205, 1992.  
  Articolo classico sull’euristica min-conflicts, utilizzata come procedura di repair locale nella pipeline.  
  🔗 [PDF](https://www.dcs.gla.ac.uk/~pat/cpM/papers/mintonAIJ.pdf)




