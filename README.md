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
- **Data**:: Our datasets are stored at [GHLAD_DataSet](https://zenodo.org/records/17219017). You can download them from this link, or use the .zip files in the project's "dataset" folder (unzip first before use).

# How to run code

## 1. Clone the Repository
Pull the GHLAD code and experimental data from the official repo (linked in the paper):
```bash
git clone https://github.com/com-prehension/GHLAD.git
```

## 2. Prepare Dataset
1. Download the dataset (link available in the paper)
2. Unzip the dataset files
3. Place the unzipped dataset folders into the project's "dataset" directory
   Required structure:
   ```bash
   GHLAD/
      dataset/
          forum/
          novel/
          halo/
   ```


## 3. Run the Code

### Basic Usage (Default Parameters)
Execute with default settings:
```bash
cd GHLAD/NODE_TRANSFORMER/
python stated_Hierarchy.py
```


### Advanced Usage (Custom Parameters)
If you need to customize parameters , use the command format below. Replace the parameter values with your actual needs:
```bash
python stated_Hierarchy.py \
    --dataname "forum" \
    --specify_data false \
    --specify_number "1" \
    --storage_number "test" \
    --project_root "/path/to/your/GHLAD" \
    --dataset_dir "/path/to/custom/dataset" \
    --result_dir "/path/to/save/results" \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --seed 42
```


### Parameter Explanations
```bash
--dataname        : Dataset name (forum/novel/halo) - default: "forum".
--specify_data    : Use specific dataset subset (true/false) - default: false.
--specify_number  : Specific dataset identifier (when specify_data=true) - default: "1".
--storage_number  : Result file identifier - default: "test".
--project_root    : Absolute path to project root - default: auto-calculated.
--dataset_root     : Custom dataset path - default: [project_root]/dataset.
--result_dir      : Custom results path - default: [project_root]/result.
--train_ratio     : Training data proportion (0-1) - default: 0.8.
--val_ratio       : Validation data proportion (0-1) - default: 0.1.
--seed            : Random seed for reproducibility - default: 42.
```


