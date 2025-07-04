# YassineTanabene-Parallel-Machine-Scheduling-Optimization

# Parallel Machine Scheduling Optimization

This repository provides an implementation of various **heuristic**, **metaheuristic**, and **exact** algorithms for solving the **parallel machine scheduling problem**. It is designed as a modular and extensible framework to facilitate experimentation and performance comparison of different solution methods.

---

## 📘 Overview

The **Parallel Machine Scheduling Problem** involves assigning a set of tasks to multiple parallel machines in such a way that the **makespan** (maximum completion time across all machines) is minimized. Each task is defined by a processing time and may include additional constraints such as mold compatibility.

---

## 📂 Code Structure

The core script is `Projet_Optimisation_Combinatoire.py`, which contains the following components:

- **Problem Definition**
  - `ParallelMachinesProblem`: Represents the scheduling problem, with methods for evaluating and validating solutions.

- **Instance Reader**
  - `InstanceReader`: Handles loading and parsing of problem instances from input files, with built-in error handling.

- **Optimization Algorithms**
  - **Heuristics**:
    - `GreedyHeuristic`
    - `LPTHeuristic` (Longest Processing Time)
    - `SPTHeuristic` (Shortest Processing Time)

  - **Metaheuristics**:
    - `RandomizedGreedyHeuristic`
    - `LocalSearch`
    - `TabuSearch`
    - `SimulatedAnnealing`
    - `GeneticAlgorithm`
    - `AntColonyOptimization`

  - **Exact Methods**:
    - `BranchAndBound`

- **Experiment Framework**
  - `Experiment`: Used to run and compare different algorithms on multiple problem instances.

- **User Interface**
  - `CLIInterface`: A command-line interface that allows users to select instances, choose algorithms, and view results interactively.

---

## 🧠 Algorithms Implemented

### Heuristics
- Greedy Heuristic
- Longest Processing Time (LPT)
- Shortest Processing Time (SPT)

### Metaheuristics
- Randomized Greedy Algorithm
- Local Search
- Tabu Search
- Simulated Annealing
- Genetic Algorithm
- Ant Colony Optimization

### Exact Methods
- Branch and Bound

---

## 📄 Instance File Format

The repository includes example problem instances. Each instance file should follow this structure:

1. **First line**: either a total number of instances or configuration parameters.
2. For each instance:
   - **Line 1**: `n_tasks n_machines instance_id other_param`
   - **Line 2**: Processing times for each task
   - **Line 3**: Mold constraints for each task (if any)

All instances are read using the `InstanceReader` class, which supports flexible formats and validates the input.

---

## 🚀 Usage

To run the script:

```bash
python Projet_Optimisation_Combinatoire.py
