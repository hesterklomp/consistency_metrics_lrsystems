# Project Title: Consistency Metrics for LR-Systems: A Comparative Analysis

## Project Overview
This project compares different metrics to evaluate the consistency of LR-systems. In particular, three metrics are considered: **Cllr^cal**, **devPAV**, and **Fid**, where **Fid** is based on a method using fiducial distributions. 
These metrics are first individually optimized using normally distributed data. 
Then, they are applied to seven datasets based on real-life LR-data, to evaluate their performance in distinguishing between consistent and inconsistent LR-systems, and their reliability across dataset type and size.

## Project Structure

### 1. **functions_thesis.py**
- Contains all functions defined for this thesis project.

### 2. **cllr test.py**
- Optimizes the **Cllr^cal** metric using normally distributed data.
- Evaluates performance of **Cllr^cal** using different scoring rules to see which one distinguishes best between consistent and inconsistent LR-systems.

### 3. **devpav test.py**
- Optimizes the **devPAV** metric using normally distributed data.
- Evaluates performance of **devPAV**, scaled in different ways, to see which one distinguishes best between consistent and inconsistent LR-systems.

### 4. **fid test.py**
- Optimizes different metrics based on the fiducial method using normally distributed data. The final metric will be called **Fid**.
- Evaluates performance of **Fid**, defined in different ways, to see which one distinguishes best between consistent and inconsistent LR-systems.
  
### 5. **data1.py to data7.py**
- Each file represents testing the three optimized metrics on a different real-life LR-dataset.
- Provides insight into how the metrics generalize across varying data structures.
- Tests how well the optimized metrics distinguish between consistent and inconsistent LR-systems based on real-life LR-data.

### 6. **all_data_metric_comparison.py**
- Compares the reliability of all three metrics (**Cllr^cal**, **devPAV**, **Fid**) across datasets.
- Reliability is tested both across different datasets as well as across different dataset sizes.
  
## Usage
1. **Optimization Phase**: Run the test files to individually optimize **Cllr^cal**, **devPAV**, and **Fid**.
2. **Distinction Test**: Run the files **data1.py** through **data7.py** to test the performance of the optimized metrics on different datasets.
3. **Reliability Test**: Run **all_data_metric_comparison.py** to compare the reliability of the optimized metrics across dataset type and size.
   
## Running the Project
1. Ensure all dependencies are installed by running:
   ```bash
   pip install -r requirements.txt
2. To test the different versions of the metrics to find which one works best, run:
   - For **Cllr^cal**:
     ```bash
     python "cllr test.py"
     ```
   - For **devPAV**:
     ```bash
     python "devpav test.py"
     ```
   - For **Fid**:
     ```bash
     python "fid test.py"
     ```
3. To test optimized metrics on different datasets, execute:
   - **Data 1 through Data 7**:
     ```bash
     python data1.py
     python data2.py
     python data3.py
     python data4.py
     python data5.py
     python data6.py
     python data7.py
     ```
4. To test the reliability of the metrics across dataset type and size, execute:
   ```bash
   python all_data_metric_comparison.py
   ```
