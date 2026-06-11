# Colorazione di grafi con GNN ed energia di Potts differenziabile

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-GNN-3C2179)
![NetworkX](https://img.shields.io/badge/NetworkX-Algoritmi%20su%20grafi-orange)
![NumPy](https://img.shields.io/badge/NumPy-Calcolo%20numerico-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Analisi%20dati-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualizzazione-11557c)
![Licenza: MIT](https://img.shields.io/badge/Licenza-MIT-green.svg)

## Descrizione

Questa repository contiene l’implementazione di una pipeline per la **colorazione di grafi** basata su **Graph Neural Network** e su una rilassazione differenziabile dell’**energia antiferromagnetica di Potts**.

L’obiettivo è assegnare un colore a ciascun nodo di un grafo non orientato, usando un numero fissato di colori `c`, in modo che due nodi adiacenti non abbiano lo stesso colore.

La pipeline combina:

- parsing di grafi in formato DIMACS;
- conversione in strutture dati compatibili con PyTorch Geometric;
- embedding trainabili dei nodi;
- message passing con connessioni residuali;
- loss differenziabile ispirata al modello di Potts;
- discretizzazione tramite `argmax`;
- post-processing con euristica locale min-conflicts;
- confronto con baseline greedy di NetworkX;
- visualizzazione dei risultati.

Gli esperimenti sono stati condotti sulle istanze DIMACS `g6` e `g12`.

## Contesto del progetto

Progetto sviluppato per il corso:

**Applicazioni Fisiche al Machine Learning**  
Sapienza Università di Roma  
A.A. 2025–2026

Autrici:

- Eleonora De Cicco
- Mariagiusi Nicodemo

## Formulazione del problema

Dato un grafo non orientato:

```math
G = (V, E)
