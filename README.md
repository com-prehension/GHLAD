# GHLAD: Graph Hierarchy-Based Log Anomaly Detection
This repository hosts the official implementation of **GHLAD** (Graph Hierarchy-based Log Anomaly Detection), corresponding to the method proposed in the paper *Log Anomaly Detection Based on Graph Hierarchy.pdf*.


## 1. Project Overview
GHLAD is a code file-level log anomaly detection framework designed to address key limitations of traditional graph-based methods:  
- Traditional methods only focus on **explicit invocation relationships** between log events/code files, ignoring inherent system hierarchies (e.g., business logic layer, data access layer) and implicit collaborative dependencies.  
- GHLAD introduces hierarchical information via deep clustering and a hierarchical message propagation mechanism to enhance node representations, enabling more accurate identification of anomalous code files.


## 2. Core Workflow
### 2.1 Log Graph Construction
Convert raw log data (containing timestamp, code file, event type, execution duration, exception info) into a directed log graph \( G=(V,E) \):  
- Nodes \( V \): Each node represents a code file in a log entry (i.e., the code file that executes the current event).  
- Edges \( E \): Directed edges indicate invocation relationships between code files (e.g., \( e(v_i,v_j) \) means \( v_i \)'s code file invokes \( v_j \)'s).

### 2.2 Key Modules
1. **Log Graph Learning**: Use Graph Transformer (TransformerConv) to learn initial structural representations of nodes, capturing explicit dependencies via attention-based neighbor aggregation.  
2. **Node Hierarchy Capturing**: Apply deep clustering (with K-means++ for center initialization) to automatically discover latent hierarchies of code files in the log graph.  
3. **Hierarchical Message Propagation**: Treat same-hierarchy nodes as temporal sequences, add position encoding, and use GRU to model implicit collaborative dependencies, enhancing node discriminability.  
4. **Anomaly Detection**: A classification layer outputs anomaly probabilities for nodes; threshold (0.5) labels nodes as anomalous (1) or normal (0) to locate faulty code files.


## 3. Experimental Validation
### 3.1 Datasets
Evaluated on 3 real-world log datasets (statistics in the paper):  
- Forum, Halo, Novel (all derived from real system logs, with anomalous graphs accounting for ~45% and anomalous nodes ranging from 3.48% to 13.24%).

### 3.2 Performance
GHLAD consistently outperforms baselines (e.g., PCA, SVM, GIN, DeepLog, SLAD) across metrics (F1-Score, Precision, Recall, PR-AUC) on all datasets, verifying its effectiveness for code file-level anomaly detection.


## 4. Resources
- **Data**: Our datasets are in the dataset folder and are in .zip format. Please unzip them before conducting experiments and use them as needed.

# How to run code

## 1. Clone the Repository
Pull the GHLAD code and experimental data from the official repo (linked in the paper):
```bash
git clone https://github.com/com-prehension/GHLAD.git
```

