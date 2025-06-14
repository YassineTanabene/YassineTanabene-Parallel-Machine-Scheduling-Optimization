import random
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os

class ParallelMachinesProblem:
    def __init__(self, n_tasks, n_machines, instance_id, other_param, processing_times, molds):
        """
        Initialise le problème des machines parallèles
        
        Args:
            n_tasks (int): Nombre de tâches
            n_machines (int): Nombre de machines
            instance_id (int): Identifiant de l'instance
            other_param (int): Autre paramètre
            processing_times (list): Durées d'exécution des tâches
            molds (list): Contraintes de mold pour chaque tâche (1 ou 2)
        """
        self.n_tasks = n_tasks
        self.n_machines = n_machines
        self.instance_id = instance_id
        self.other_param = other_param
        self.processing_times = processing_times
        self.molds = molds
        
        # Créer une liste de tâches avec leur indice, durée et mold
        self.tasks = [(i, processing_times[i], molds[i]) for i in range(n_tasks)]
    
    def evaluate_solution(self, assignment):
        """
        Évalue une solution donnée en calculant le makespan
        
        Args:
            assignment (list): Liste indiquant quelle machine (0 ou 1) traite chaque tâche
            
        Returns:
            int: Le makespan (temps maximal d'exécution entre les machines)
        """
        if len(assignment) != self.n_tasks:
            raise ValueError("La taille de l'affectation doit être égale au nombre de tâches")
        
        # Calculer la charge de travail pour chaque machine
        machine_loads = [0] * self.n_machines
        
        for task_idx, machine in enumerate(assignment):
            if machine < 0 or machine >= self.n_machines:
                raise ValueError(f"Machine invalide: {machine}")
            
            machine_loads[machine] += self.processing_times[task_idx]
        
        # Le makespan est le temps de la machine la plus chargée
        return max(machine_loads)
    
    def is_valid_solution(self, assignment):
        """
        Vérifie si une solution respecte les contraintes de mold
        
        Args:
            assignment (list): Liste indiquant quelle machine (0 ou 1) traite chaque tâche
            
        Returns:
            bool: True si la solution est valide, False sinon
        """
        # Vérifier que chaque tâche est affectée à une machine valide
        if not all(0 <= machine < self.n_machines for machine in assignment):
            return False
        
        # Regrouper les tâches par mold et par machine
        mold_machine_tasks = defaultdict(list)
        
        for task_idx, machine in enumerate(assignment):
            mold = self.molds[task_idx]
            mold_machine_tasks[(mold, machine)].append(task_idx)
        
        # Pour chaque mold, vérifier les contraintes spécifiques
        # Note: Dans cette implémentation, nous supposons qu'il n'y a pas de contraintes
        # spécifiques entre les molds autres que leur attribution à des machines
        
        return True


class InstanceReader:
    @staticmethod
    def read_instances(file_path):
        """
        Lit les instances à partir d'un fichier
        
        Args:
            file_path (str): Chemin vers le fichier d'instances
            
        Returns:
            list: Liste des problèmes
        """
        problems = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Vérifier si le fichier est vide
                if not lines:
                    print(f"Erreur: Le fichier {file_path} est vide.")
                    return problems
                
                # Afficher le début du fichier pour débogage
                print(f"Lecture du fichier {file_path}")
                print(f"Premières lignes du fichier:")
                for i in range(min(5, len(lines))):
                    print(f"  {i}: {lines[i].strip()}")
                
                try:
                    # La première ligne peut être le nombre total d'instances ou la première instance
                    first_line = lines[0].strip()
                    parts = first_line.split()
                    
                    # Vérifier si la première ligne est un seul nombre
                    if len(parts) == 1 and parts[0].isdigit():
                        print(f"La première ligne contient le nombre total d'instances: {parts[0]}")
                        line_idx = 1  # Commencer à la deuxième ligne
                    else:
                        print("La première ligne ne contient pas le nombre total d'instances, mais les données de la première instance")
                        line_idx = 0  # Commencer à la première ligne
                    
                    while line_idx < len(lines):
                        # S'assurer que la ligne n'est pas vide
                        if not lines[line_idx].strip():
                            line_idx += 1
                            continue
                        
                        # Lire les paramètres de l'instance
                        params = list(map(int, lines[line_idx].strip().split()))
                        if len(params) < 4:
                            print(f"Erreur: Format incorrect à la ligne {line_idx}: {lines[line_idx].strip()}")
                            print(f"Nombre de paramètres trouvés: {len(params)}, attendus: 4")
                            line_idx += 1
                            continue
                        
                        n_tasks, n_machines, instance_id, other_param = params
                        line_idx += 1
                        
                        # Vérifier que nous avons assez de lignes
                        if line_idx >= len(lines) or line_idx + 1 >= len(lines):
                            print("Erreur: Pas assez de lignes dans le fichier")
                            break
                        
                        # Lire les durées d'exécution
                        processing_times_line = lines[line_idx].strip()
                        processing_times = list(map(int, processing_times_line.split()))
                        line_idx += 1
                        
                        # Vérifier la cohérence des données
                        if len(processing_times) != n_tasks:
                            print(f"Avertissement: Le nombre de durées ({len(processing_times)}) ne correspond pas au nombre de tâches ({n_tasks})")
                            print(f"  Ligne des durées: {processing_times_line}")
                            # Ajuster le nombre de tâches si nécessaire
                            n_tasks = len(processing_times)
                        
                        # Lire les contraintes de mold
                        molds_line = lines[line_idx].strip()
                        molds = list(map(int, molds_line.split()))
                        line_idx += 1
                        
                        # Vérifier la cohérence des données pour les molds
                        if len(molds) != n_tasks:
                            print(f"Avertissement: Le nombre de molds ({len(molds)}) ne correspond pas au nombre de tâches ({n_tasks})")
                            print(f"  Ligne des molds: {molds_line}")
                            # Ajuster pour avoir la même longueur
                            if len(molds) < n_tasks:
                                molds.extend([1] * (n_tasks - len(molds)))
                            else:
                                molds = molds[:n_tasks]
                        
                        # Créer et ajouter le problème
                        problem = ParallelMachinesProblem(
                            n_tasks, n_machines, instance_id, other_param,
                            processing_times, molds
                        )
                        problems.append(problem)
                        
                except Exception as e:
                    print(f"Erreur lors du traitement des données: {e}")
                    import traceback
                    traceback.print_exc()
                
        except FileNotFoundError:
            print(f"Erreur: Le fichier {file_path} n'a pas été trouvé.")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Nombre d'instances lues: {len(problems)}")
        return problems


class GreedyHeuristic:
    @staticmethod
    def solve(problem):
        """
        Résout le problème en utilisant une heuristique gloutonne simple:
        trier les tâches par ordre décroissant de durée et les affecter à la machine la moins chargée
        
        Args:
            problem (ParallelMachinesProblem): Le problème à résoudre
            
        Returns:
            tuple: (assignment, makespan)
        """
        # Trier les tâches par ordre décroissant de durée
        sorted_tasks = sorted(range(problem.n_tasks), key=lambda i: problem.processing_times[i], reverse=True)
        
        # Initialiser l'affectation et les charges des machines
        assignment = [-1] * problem.n_tasks
        machine_loads = [0] * problem.n_machines
        
        # Affecter chaque tâche à la machine la moins chargée
        for task_idx in sorted_tasks:
            # Trouver la machine la moins chargée
            min_load_machine = machine_loads.index(min(machine_loads))
            
            # Affecter la tâche à cette machine
            assignment[task_idx] = min_load_machine
            machine_loads[min_load_machine] += problem.processing_times[task_idx]
        
        # Calculer le makespan
        makespan = max(machine_loads)
        
        return assignment, makespan


class RandomizedGreedyHeuristic:
    @staticmethod
    def solve(problem, alpha=0.3):
        """
        Résout le problème en utilisant une heuristique gloutonne randomisée:
        à chaque étape, choisit aléatoirement une tâche parmi les alpha% plus grandes
        
        Args:
            problem (ParallelMachinesProblem): Le problème à résoudre
            alpha (float): Paramètre de randomisation (0 <= alpha <= 1)
            
        Returns:
            tuple: (assignment, makespan)
        """
        # Trier les tâches par ordre décroissant de durée
        sorted_tasks = sorted(range(problem.n_tasks), key=lambda i: problem.processing_times[i], reverse=True)
        
        # Initialiser l'affectation et les charges des machines
        assignment = [-1] * problem.n_tasks
        machine_loads = [0] * problem.n_machines
        
        # Liste des tâches non affectées
        unassigned_tasks = sorted_tasks.copy()
        
        while unassigned_tasks:
            # Déterminer la taille de la liste restreinte de candidats (RCL)
            rcl_size = max(1, int(alpha * len(unassigned_tasks)))
            
            # Sélectionner aléatoirement une tâche parmi les rcl_size plus grandes
            task_idx = unassigned_tasks.pop(random.randint(0, rcl_size - 1))
            
            # Trouver la machine la moins chargée
            min_load_machine = machine_loads.index(min(machine_loads))
            
            # Affecter la tâche à cette machine
            assignment[task_idx] = min_load_machine
            machine_loads[min_load_machine] += problem.processing_times[task_idx]
        
        # Calculer le makespan
        makespan = max(machine_loads)
        
        return assignment, makespan


class LocalSearch:
    @staticmethod
    def solve(problem, initial_solution=None, max_iterations=1000):
        """
        Résout le problème en utilisant une recherche locale simple
        
        Args:
            problem (ParallelMachinesProblem): Le problème à résoudre
            initial_solution (list): Solution initiale (si None, utilise l'heuristique gloutonne)
            max_iterations (int): Nombre maximum d'itérations
            
        Returns:
            tuple: (assignment, makespan)
        """
        # Obtenir une solution initiale si non fournie
        if initial_solution is None:
            initial_solution, _ = GreedyHeuristic.solve(problem)
        
        current_solution = initial_solution.copy()
        current_makespan = problem.evaluate_solution(current_solution)
        
        for _ in range(max_iterations):
            # Choisir deux tâches aléatoires
            task1 = random.randint(0, problem.n_tasks - 1)
            task2 = random.randint(0, problem.n_tasks - 1)
            
            if task1 == task2:
                continue
            
            # Échanger les machines des deux tâches
            new_solution = current_solution.copy()
            new_solution[task1], new_solution[task2] = new_solution[task2], new_solution[task1]
            
            # Évaluer la nouvelle solution
            if problem.is_valid_solution(new_solution):
                new_makespan = problem.evaluate_solution(new_solution)
                
                # Accepter la nouvelle solution si elle est meilleure
                if new_makespan < current_makespan:
                    current_solution = new_solution
                    current_makespan = new_makespan
        
        return current_solution, current_makespan


class TabuSearch:
    @staticmethod
    def solve(problem, initial_solution=None, max_iterations=1000, tabu_tenure=10):
        """
        Résout le problème en utilisant la recherche tabou
        
        Args:
            problem (ParallelMachinesProblem): Le problème à résoudre
            initial_solution (list): Solution initiale (si None, utilise l'heuristique gloutonne)
            max_iterations (int): Nombre maximum d'itérations
            tabu_tenure (int): Durée tabou (nombre d'itérations pendant lesquelles un mouvement est interdit)
            
        Returns:
            tuple: (assignment, makespan)
        """
        # Obtenir une solution initiale si non fournie
        if initial_solution is None:
            initial_solution, _ = GreedyHeuristic.solve(problem)
        
        current_solution = initial_solution.copy()
        current_makespan = problem.evaluate_solution(current_solution)
        
        best_solution = current_solution.copy()
        best_makespan = current_makespan
        
        # Liste tabou (paires de tâches qui ne peuvent pas être échangées)
        tabu_list = {}
        iteration = 0
        
        for iteration in range(max_iterations):
            best_neighbor = None
            best_neighbor_makespan = float('inf')
            best_move = None
            
            # Explorer tous les voisins (échanges de paires de tâches)
            for task1 in range(problem.n_tasks):
                for task2 in range(task1 + 1, problem.n_tasks):
                    # Vérifier si le mouvement est tabou
                    if (task1, task2) in tabu_list and tabu_list[(task1, task2)] > iteration:
                        continue
                    
                    # Échanger les machines des deux tâches
                    new_solution = current_solution.copy()
                    new_solution[task1], new_solution[task2] = new_solution[task2], new_solution[task1]
                    
                    # Évaluer la nouvelle solution
                    if problem.is_valid_solution(new_solution):
                        new_makespan = problem.evaluate_solution(new_solution)
                        
                        # Mettre à jour le meilleur voisin
                        if new_makespan < best_neighbor_makespan:
                            best_neighbor = new_solution
                            best_neighbor_makespan = new_makespan
                            best_move = (task1, task2)
            
            # S'il n'y a pas de voisin valide, arrêter
            if best_neighbor is None:
                break
            
            # Mettre à jour la solution courante
            current_solution = best_neighbor
            current_makespan = best_neighbor_makespan
            
            # Ajouter le mouvement à la liste tabou
            tabu_list[best_move] = iteration + tabu_tenure
            
            # Mettre à jour la meilleure solution
            if current_makespan < best_makespan:
                best_solution = current_solution.copy()
                best_makespan = current_makespan
        
        return best_solution, best_makespan


class SimulatedAnnealing:
    @staticmethod
    def solve(problem, initial_solution=None, initial_temp=100, final_temp=1, cooling_rate=0.95, iterations_per_temp=100):
        """
        Résout le problème en utilisant le recuit simulé
        
        Args:
            problem (ParallelMachinesProblem): Le problème à résoudre
            initial_solution (list): Solution initiale (si None, utilise l'heuristique gloutonne)
            initial_temp (float): Température initiale
            final_temp (float): Température finale
            cooling_rate (float): Taux de refroidissement
            iterations_per_temp (int): Nombre d'itérations par température
            
        Returns:
            tuple: (assignment, makespan)
        """
        # Obtenir une solution initiale si non fournie
        if initial_solution is None:
            initial_solution, _ = GreedyHeuristic.solve(problem)
        
        current_solution = initial_solution.copy()
        current_makespan = problem.evaluate_solution(current_solution)
        
        best_solution = current_solution.copy()
        best_makespan = current_makespan
        
        temperature = initial_temp
        
        while temperature > final_temp:
            for _ in range(iterations_per_temp):
                # Choisir deux tâches aléatoires
                task1 = random.randint(0, problem.n_tasks - 1)
                task2 = random.randint(0, problem.n_tasks - 1)
                
                if task1 == task2:
                    continue
                
                # Échanger les machines des deux tâches
                new_solution = current_solution.copy()
                new_solution[task1], new_solution[task2] = new_solution[task2], new_solution[task1]
                
                # Évaluer la nouvelle solution
                if problem.is_valid_solution(new_solution):
                    new_makespan = problem.evaluate_solution(new_solution)
                    
                    # Calculer la différence de makespan
                    delta = new_makespan - current_makespan
                    
                    # Accepter la nouvelle solution selon la probabilité
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_solution = new_solution
                        current_makespan = new_makespan
                        
                        # Mettre à jour la meilleure solution
                        if current_makespan < best_makespan:
                            best_solution = current_solution.copy()
                            best_makespan = current_makespan
            
            # Refroidir la température
            temperature *= cooling_rate
        
        return best_solution, best_makespan


class GeneticAlgorithm:
    @staticmethod
    def solve(problem, population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.2):
        """
        Résout le problème en utilisant un algorithme génétique
        
        Args:
            problem (ParallelMachinesProblem): Le problème à résoudre
            population_size (int): Taille de la population
            generations (int): Nombre de générations
            crossover_rate (float): Taux de croisement
            mutation_rate (float): Taux de mutation
            
        Returns:
            tuple: (assignment, makespan)
        """
        # Fonction pour créer un individu aléatoire
        def create_random_individual():
            return [random.randint(0, problem.n_machines - 1) for _ in range(problem.n_tasks)]
        
        # Fonction de fitness (à minimiser)
        def fitness(individual):
            if not problem.is_valid_solution(individual):
                return float('inf')
            return problem.evaluate_solution(individual)
        
        # Fonction de croisement (one-point crossover)
        def crossover(parent1, parent2):
            if random.random() > crossover_rate:
                return parent1.copy()
            
            crossover_point = random.randint(1, problem.n_tasks - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            return child
        
        # Fonction de mutation
        def mutate(individual):
            mutated = individual.copy()
            for i in range(problem.n_tasks):
                if random.random() < mutation_rate:
                    mutated[i] = random.randint(0, problem.n_machines - 1)
            return mutated
        
        # Initialisation de la population
        population = [create_random_individual() for _ in range(population_size)]
        
        # Évaluation initiale
        fitnesses = [fitness(ind) for ind in population]
        
        # Meilleur individu trouvé
        best_individual = population[fitnesses.index(min(fitnesses))]
        best_fitness = min(fitnesses)
        
        for _ in range(generations):
            # Sélection par tournoi
            new_population = []
            
            for _ in range(population_size):
                # Sélectionner deux parents par tournoi
                tournament_size = 3
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                parent1_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
                parent1 = population[parent1_idx]
                
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                parent2_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
                parent2 = population[parent2_idx]
                
                # Croisement
                child = crossover(parent1, parent2)
                
                # Mutation
                child = mutate(child)
                
                new_population.append(child)
            
            # Mise à jour de la population
            population = new_population
            
            # Évaluation
            fitnesses = [fitness(ind) for ind in population]
            
            # Mise à jour du meilleur individu
            if min(fitnesses) < best_fitness:
                best_fitness = min(fitnesses)
                best_individual = population[fitnesses.index(best_fitness)]
        
        return best_individual, best_fitness


class BranchAndBound:
    @staticmethod
    def solve(problem, time_limit=60):
        """
        Résout le problème en utilisant Branch and Bound
        
        Args:
            problem (ParallelMachinesProblem): Le problème à résoudre
            time_limit (int): Limite de temps en secondes
            
        Returns:
            tuple: (assignment, makespan) ou (None, None) si le temps est dépassé
        """
        # Obtenir une borne supérieure avec une heuristique
        _, upper_bound = GreedyHeuristic.solve(problem)
        
        # Calculer une borne inférieure simple (somme des durées / nombre de machines)
        lower_bound = sum(problem.processing_times) // problem.n_machines
        if sum(problem.processing_times) % problem.n_machines != 0:
            lower_bound += 1
        
        # Autre borne inférieure: la durée de la tâche la plus longue
        max_task_duration = max(problem.processing_times)
        lower_bound = max(lower_bound, max_task_duration)
        
        # Initialiser la meilleure solution
        best_solution = None
        best_makespan = upper_bound
        
        # Fonction récursive pour explorer l'arbre de recherche
        def branch_and_bound_recursive(assigned, machine_loads, level, start_time):
            nonlocal best_solution, best_makespan
            
            # Vérifier si le temps est dépassé
            if time.time() - start_time > time_limit:
                return False  # Temps dépassé
            
            # Si toutes les tâches sont affectées
            if level == problem.n_tasks:
                # Calculer le makespan
                makespan = max(machine_loads)
                
                # Mettre à jour la meilleure solution
                if makespan < best_makespan:
                    best_solution = assigned.copy()
                    best_makespan = makespan
                
                return True
            
            # Calculer une borne inférieure pour cette branche
            current_lower_bound = max(max(machine_loads), (sum(problem.processing_times[level:]) + sum(machine_loads)) // problem.n_machines)
            
            # Élagage
            if current_lower_bound >= best_makespan:
                return True  # Ne pas explorer cette branche
            
            # Explorer chaque machine pour la tâche actuelle
            for machine in range(problem.n_machines):
                # Affecter la tâche à la machine
                assigned[level] = machine
                machine_loads[machine] += problem.processing_times[level]
                
                # Explorer la branche
                branch_and_bound_recursive(assigned, machine_loads, level + 1, start_time)
                
                # Annuler l'affectation (backtracking)
                machine_loads[machine] -= problem.processing_times[level]
            
            return True
        
        # Lancer la recherche
        start_time = time.time()
        assigned = [-1] * problem.n_tasks
        machine_loads = [0] * problem.n_machines
        
        success = branch_and_bound_recursive(assigned, machine_loads, 0, start_time)
        
        if not success or best_solution is None:
            return None, None  # Temps dépassé
        
        return best_solution, best_makespan


class LPTHeuristic:
    @staticmethod
    def solve(problem):
        """
        Résout le problème en utilisant l'heuristique LPT (Longest Processing Time First):
        trier les tâches par ordre décroissant de durée et les affecter à la machine la moins chargée
        
        Args:
            problem (ParallelMachinesProblem): Le problème à résoudre
            
        Returns:
            tuple: (assignment, makespan)
        """
        # Trier les tâches par ordre décroissant de durée
        sorted_tasks = sorted(range(problem.n_tasks), key=lambda i: problem.processing_times[i], reverse=True)
        
        # Initialiser l'affectation et les charges des machines
        assignment = [-1] * problem.n_tasks
        machine_loads = [0] * problem.n_machines
        
        # Affecter chaque tâche à la machine la moins chargée
        for task_idx in sorted_tasks:
            # Trouver la machine la moins chargée
            min_load_machine = machine_loads.index(min(machine_loads))
            
            # Affecter la tâche à cette machine
            assignment[task_idx] = min_load_machine
            machine_loads[min_load_machine] += problem.processing_times[task_idx]
        
        # Calculer le makespan
        makespan = max(machine_loads)
        
        return assignment, makespan


class SPTHeuristic:
    @staticmethod
    def solve(problem):
        """
        Résout le problème en utilisant l'heuristique SPT (Shortest Processing Time First):
        trier les tâches par ordre croissant de durée et les affecter à la machine la moins chargée
        
        Args:
            problem (ParallelMachinesProblem): Le problème à résoudre
            
        Returns:
            tuple: (assignment, makespan)
        """
        # Trier les tâches par ordre croissant de durée
        sorted_tasks = sorted(range(problem.n_tasks), key=lambda i: problem.processing_times[i])
        
        # Initialiser l'affectation et les charges des machines
        assignment = [-1] * problem.n_tasks
        machine_loads = [0] * problem.n_machines
        
        # Affecter chaque tâche à la machine la moins chargée
        for task_idx in sorted_tasks:
            # Trouver la machine la moins chargée
            min_load_machine = machine_loads.index(min(machine_loads))
            
            # Affecter la tâche à cette machine
            assignment[task_idx] = min_load_machine
            machine_loads[min_load_machine] += problem.processing_times[task_idx]
        
        # Calculer le makespan
        makespan = max(machine_loads)
        
        return assignment, makespan

class AntColonyOptimization:
    @staticmethod
    def solve(problem, n_ants=10, n_iterations=50, alpha=1.0, beta=2.0, rho=0.5, q0=0.9):
        """
        Résout le problème en utilisant l'optimisation par colonie de fourmis
        
        Args:
            problem (ParallelMachinesProblem): Le problème à résoudre
            n_ants (int): Nombre de fourmis
            n_iterations (int): Nombre d'itérations
            alpha (float): Importance des phéromones
            beta (float): Importance de l'heuristique
            rho (float): Taux d'évaporation des phéromones
            q0 (float): Paramètre d'exploitation vs exploration (0<=q0<=1)
            
        Returns:
            tuple: (assignment, makespan)
        """
        import numpy as np
        
        # Initialiser les phéromones
        # Utilisation d'une matrice de phéromones tâche x machine
        pheromones = np.ones((problem.n_tasks, problem.n_machines))
        
        # Initialiser la meilleure solution
        best_assignment = None
        best_makespan = float('inf')
        
        # Définir l'heuristique (inverse de la durée d'exécution)
        # Chaque tâche a la même valeur heuristique pour toutes les machines
        heuristic = np.array([[1.0 / problem.processing_times[i]] * problem.n_machines for i in range(problem.n_tasks)])
        
        for _ in range(n_iterations):
            # Pour chaque fourmi
            for _ in range(n_ants):
                # Initialiser l'affectation et les charges des machines
                assignment = [-1] * problem.n_tasks
                machine_loads = [0] * problem.n_machines
                
                # Ordre aléatoire des tâches
                tasks = list(range(problem.n_tasks))
                random.shuffle(tasks)
                
                # Affecter les tâches une par une
                for task_idx in tasks:
                    # Choisir une machine pour la tâche
                    if random.random() < q0:
                        # Exploitation: choisir la meilleure machine
                        machine_scores = []
                        for machine in range(problem.n_machines):
                            score = (pheromones[task_idx, machine] ** alpha) * (heuristic[task_idx, machine] ** beta)
                            machine_scores.append(score)
                        
                        # Choisir la machine avec le meilleur score
                        chosen_machine = machine_scores.index(max(machine_scores))
                    else:
                        # Exploration: choisir une machine selon une probabilité
                        probabilities = []
                        for machine in range(problem.n_machines):
                            score = (pheromones[task_idx, machine] ** alpha) * (heuristic[task_idx, machine] ** beta)
                            probabilities.append(score)
                        
                        # Normaliser les probabilités
                        total = sum(probabilities)
                        if total > 0:
                            probabilities = [p / total for p in probabilities]
                        else:
                            probabilities = [1.0 / problem.n_machines] * problem.n_machines
                        
                        # Choisir une machine selon la probabilité
                        chosen_machine = random.choices(range(problem.n_machines), weights=probabilities)[0]
                    
                    # Affecter la tâche à la machine choisie
                    assignment[task_idx] = chosen_machine
                    machine_loads[chosen_machine] += problem.processing_times[task_idx]
                
                # Évaluer la solution
                makespan = max(machine_loads)
                
                # Mettre à jour la meilleure solution
                if makespan < best_makespan:
                    best_assignment = assignment.copy()
                    best_makespan = makespan
                
                # Mettre à jour les phéromones localement
                for task_idx in range(problem.n_tasks):
                    machine = assignment[task_idx]
                    pheromones[task_idx, machine] = (1 - rho) * pheromones[task_idx, machine] + rho * (1.0 / makespan)
            
            # Évaporation globale des phéromones
            pheromones = (1 - rho) * pheromones
            
            # Renforcer les phéromones sur le meilleur chemin
            if best_assignment is not None:
                for task_idx in range(problem.n_tasks):
                    machine = best_assignment[task_idx]
                    pheromones[task_idx, machine] += rho * (1.0 / best_makespan)
        
        return best_assignment, best_makespan


class Experiment:
    """
    Classe pour gérer les expériences avec différentes méthodes de résolution
    et générer des statistiques
    """
    
    def __init__(self):
        # Définir les méthodes de résolution disponibles
        self.heuristics = {
            "Greedy": GreedyHeuristic.solve,
            "LPT": LPTHeuristic.solve,
            "SPT": SPTHeuristic.solve
        }
        
        self.metaheuristics = {
            "Randomized Greedy (α=0.3)": lambda p: RandomizedGreedyHeuristic.solve(p, alpha=0.3),
            "Local Search": lambda p: LocalSearch.solve(p, max_iterations=1000),
            "TabuSearch": lambda p: TabuSearch.solve(p, max_iterations=100, tabu_tenure=10),
            "Simulated Annealing": lambda p: SimulatedAnnealing.solve(p, iterations_per_temp=50),
            "Genetic Algorithm": lambda p: GeneticAlgorithm.solve(p, generations=50),
            "Ant Colony Optimization": lambda p: AntColonyOptimization.solve(p, n_iterations=30)
        }
        
        self.exact_methods = {
            "Branch and Bound": lambda p: BranchAndBound.solve(p, time_limit=60)
        }
        
        # Combiner toutes les méthodes
        self.all_methods = {}
        self.all_methods.update(self.heuristics)
        self.all_methods.update(self.metaheuristics)
        self.all_methods.update(self.exact_methods)
    
    def run_method_on_instance(self, method_name, problem):
        """
        Exécute une méthode sur une instance et retourne les résultats
        
        Args:
            method_name (str): Nom de la méthode
            problem (ParallelMachinesProblem): Instance du problème
            
        Returns:
            dict: Résultats incluant le temps d'exécution, le makespan, etc.
        """
        method = self.all_methods[method_name]
        
        start_time = time.time()
        assignment, makespan = method(problem)
        execution_time = time.time() - start_time
        
        # Vérifier la validité de la solution
        valid = False
        if assignment is not None:
            valid = problem.is_valid_solution(assignment)
        
        # Préparer les résultats
        results = {
            "Method": method_name,
            "Instance ID": problem.instance_id,
            "Num Tasks": problem.n_tasks,
            "Num Machines": problem.n_machines,
            "Makespan": makespan if makespan is not None else "N/A",
            "Execution Time (s)": execution_time,
            "Valid Solution": valid
        }
        
        return results
    
    def run_all_methods_on_instance(self, problem):
        """
        Exécute toutes les méthodes sur une instance et retourne les résultats
        
        Args:
            problem (ParallelMachinesProblem): Instance du problème
            
        Returns:
            list: Liste des résultats pour chaque méthode
        """
        results = []
        
        for method_name in self.all_methods:
            result = self.run_method_on_instance(method_name, problem)
            results.append(result)
        
        return results
    
    def run_method_on_all_instances(self, method_name, problems):
        """
        Exécute une méthode sur toutes les instances et retourne les résultats
        
        Args:
            method_name (str): Nom de la méthode
            problems (list): Liste d'instances du problème
            
        Returns:
            list: Liste des résultats pour chaque instance
        """
        results = []
        
        for problem in problems:
            result = self.run_method_on_instance(method_name, problem)
            results.append(result)
        
        return results
    
    def run_all_methods_on_all_instances(self, problems):
        """
        Exécute toutes les méthodes sur toutes les instances et retourne les résultats
        
        Args:
            problems (list): Liste d'instances du problème
            
        Returns:
            list: Liste des résultats
        """
        results = []
        
        for problem in problems:
            instance_results = self.run_all_methods_on_instance(problem)
            results.extend(instance_results)
        
        return results
    
    def get_instance_details(self, problem):
        """
        Retourne les détails d'une instance
        
        Args:
            problem (ParallelMachinesProblem): Instance du problème
            
        Returns:
            dict: Détails de l'instance
        """
        details = {
            "Instance ID": problem.instance_id,
            "Number of Tasks": problem.n_tasks,
            "Number of Machines": problem.n_machines,
            "Other Parameter": problem.other_param,
            "Processing Times": problem.processing_times,
            "Molds": problem.molds
        }
        
        return details
    
    def save_results_to_excel(self, results, filename="results.xlsx"):
        """
        Sauvegarde les résultats dans un fichier Excel
        
        Args:
            results (list): Liste des résultats
            filename (str): Nom du fichier Excel
        """
        # Créer un DataFrame pandas
        df = pd.DataFrame(results)
        
        # Convertir 'N/A' en valeur NaN pour éviter les erreurs de conversion
        if 'Makespan' in df.columns:
            df['Makespan'] = pd.to_numeric(df['Makespan'], errors='coerce')
        
        # Créer un writer Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Tous les résultats
            df.to_excel(writer, sheet_name='All Results', index=False)
            
            # Statistiques par heuristique
            heuristics_df = df[df['Method'].isin(self.heuristics.keys())]
            if not heuristics_df.empty:
                try:
                    pivot_heur = pd.pivot_table(heuristics_df, 
                                            values=['Makespan', 'Execution Time (s)'],
                                            index=['Instance ID'],
                                            columns=['Method'],
                                            aggfunc='mean')
                    pivot_heur.to_excel(writer, sheet_name='Heuristics Comparison')
                except Exception as e:
                    print(f"Erreur lors de la création du pivot pour les heuristiques: {e}")
            
            # Statistiques par métaheuristique
            metaheuristics_df = df[df['Method'].isin(self.metaheuristics.keys())]
            if not metaheuristics_df.empty:
                try:
                    pivot_meta = pd.pivot_table(metaheuristics_df, 
                                            values=['Makespan', 'Execution Time (s)'],
                                            index=['Instance ID'],
                                            columns=['Method'],
                                            aggfunc='mean')
                    pivot_meta.to_excel(writer, sheet_name='Metaheuristics Comparison')
                except Exception as e:
                    print(f"Erreur lors de la création du pivot pour les métaheuristiques: {e}")
            
            # Statistiques par méthode exacte
            exact_df = df[df['Method'].isin(self.exact_methods.keys())]
            if not exact_df.empty:
                try:
                    # Créer une copie pour éviter le SettingWithCopyWarning
                    exact_df = exact_df.copy()
                    
                    # Assurez-vous que la colonne Makespan est numérique
                    exact_df['Makespan'] = pd.to_numeric(exact_df['Makespan'], errors='coerce')
                    
                    pivot_exact = pd.pivot_table(exact_df, 
                                            values=['Makespan', 'Execution Time (s)'],
                                            index=['Instance ID'],
                                            columns=['Method'],
                                            aggfunc='mean')
                    pivot_exact.to_excel(writer, sheet_name='Exact Methods')
                except Exception as e:
                    print(f"Erreur lors de la création du pivot pour les méthodes exactes: {e}")
        
        print(f"Résultats sauvegardés dans {filename}")
    def generate_summary_statistics(self, results, filename="summary.xlsx"):
        """
        Génère des statistiques résumées des résultats
        
        Args:
            results (list): Liste des résultats
            filename (str): Nom du fichier Excel
        """
        df = pd.DataFrame(results)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Statistiques globales
            summary = df.groupby('Method').agg({
                'Makespan': ['mean', 'min', 'max', 'std'],
                'Execution Time (s)': ['mean', 'min', 'max', 'std'],
                'Valid Solution': 'mean'
            })
            summary.to_excel(writer, sheet_name='Global Summary')
            
            # Performance par taille d'instance
            size_perf = df.groupby(['Method', 'Num Tasks']).agg({
                'Makespan': 'mean',
                'Execution Time (s)': 'mean'
            })
            size_perf.to_excel(writer, sheet_name='Performance by Size')
            
            # Créer des graphiques de comparaison
            try:
                # Temps d'exécution moyen par méthode
                exec_time_fig, exec_time_ax = plt.subplots(figsize=(12, 6))
                avg_times = df.groupby('Method')['Execution Time (s)'].mean().sort_values()
                avg_times.plot(kind='bar', ax=exec_time_ax)
                exec_time_ax.set_title('Temps d\'exécution moyen par méthode')
                exec_time_ax.set_ylabel('Temps (s)')
                plt.tight_layout()
                
                # Makespan relatif (normalisé par rapport au minimum par instance)
                df_pivot = df.pivot_table(index='Instance ID', columns='Method', values='Makespan')
                df_norm = df_pivot.div(df_pivot.min(axis=1), axis=0)
                
                rel_makespan_fig, rel_makespan_ax = plt.subplots(figsize=(12, 6))
                df_norm.mean().sort_values().plot(kind='bar', ax=rel_makespan_ax)
                rel_makespan_ax.set_title('Makespan relatif par méthode (par rapport au minimum)')
                rel_makespan_ax.set_ylabel('Ratio')
                plt.tight_layout()
                
                # Sauvegarder les graphiques
                exec_time_fig.savefig('execution_time.png')
                rel_makespan_fig.savefig('relative_makespan.png')
                
                # Ajouter une feuille pour indiquer les graphiques sauvegardés
                graphs_info = pd.DataFrame({
                    'Graph': ['Temps d\'exécution moyen', 'Makespan relatif'],
                    'Filename': ['execution_time.png', 'relative_makespan.png']
                })
                graphs_info.to_excel(writer, sheet_name='Graphs Info', index=False)
                
            except Exception as e:
                print(f"Erreur lors de la création des graphiques: {e}")
        
        print(f"Statistiques résumées sauvegardées dans {filename}")


class CLIInterface:
    """
    Interface en ligne de commande pour l'expérimentation
    """
    
    def __init__(self):
        self.experiment = Experiment()
        self.instances_file = None
        self.problems = []
    
    def display_menu(self):
        """Affiche le menu principal"""
        print("\n" + "=" * 60)
        print(" SYSTÈME D'EXPÉRIMENTATION POUR LE PROBLÈME DES MACHINES PARALLÈLES ")
        print("=" * 60)
        print("1. Exécuter toutes les méthodes sur toutes les instances")
        print("2. Sélectionner une méthode pour toutes les instances")
        print("3. Comparer les méthodes sur une instance spécifique")
        print("4. Travailler une méthode sur une instance spécifique")
        print("5. Afficher les détails d'une instance")
        print("6. Charger un nouveau fichier d'instances")
        print("7. Quitter")
        print("=" * 60)
    
    def load_instances(self):
        """Charge les instances à partir d'un fichier"""
        file_path = input("Entrez le chemin du fichier d'instances: ")
        self.instances_file = file_path
        
        try:
            self.problems = InstanceReader.read_instances(file_path)
            if not self.problems:
                print("Aucune instance n'a été chargée. Veuillez vérifier le fichier.")
                return False
            
            print(f"{len(self.problems)} instances chargées avec succès.")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des instances: {e}")
            return False
    
    def select_instance(self):
        """Permet à l'utilisateur de sélectionner une instance"""
        if not self.problems:
            print("Aucune instance disponible. Veuillez d'abord charger un fichier d'instances.")
            return None
        
        print("\nInstances disponibles:")
        for i, problem in enumerate(self.problems):
            print(f"{i+1}. Instance {problem.instance_id} ({problem.n_tasks} tâches, {problem.n_machines} machines)")
        
        try:
            choice = int(input("\nSélectionnez une instance (numéro): "))
            if 1 <= choice <= len(self.problems):
                # Assurez-vous que l'instance sélectionnée a le bon ID correspondant à son indice
                selected_problem = self.problems[choice-1]
                # Cette ligne est essentielle - mettre à jour l'ID de l'instance avec l'index sélectionné
                # si l'ID n'est pas déjà défini correctement dans le fichier d'entrée
                if selected_problem.instance_id != choice:
                    selected_problem.instance_id = choice
                return selected_problem
            else:
                print("Choix invalide.")
                return None
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre.")
            return None
    def select_method(self):
        """Permet à l'utilisateur de sélectionner une méthode"""
        print("\nMéthodes disponibles:")
        print("\nHeuristiques:")
        for i, method in enumerate(self.experiment.heuristics.keys()):
            print(f"{i+1}. {method}")
        
        print("\nMétaheuristiques:")
        offset = len(self.experiment.heuristics)
        for i, method in enumerate(self.experiment.metaheuristics.keys()):
            print(f"{i+offset+1}. {method}")
        
        print("\nMéthodes exactes:")
        offset += len(self.experiment.metaheuristics)
        for i, method in enumerate(self.experiment.exact_methods.keys()):
            print(f"{i+offset+1}. {method}")
        
        try:
            choice = int(input("\nSélectionnez une méthode (numéro): "))
            all_methods = list(self.experiment.all_methods.keys())
            if 1 <= choice <= len(all_methods):
                return all_methods[choice-1]
            else:
                print("Choix invalide.")
                return None
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre.")
            return None
    
    def run_all_methods_all_instances(self):
        """Exécute toutes les méthodes sur toutes les instances"""
        if not self.problems:
            print("Aucune instance disponible. Veuillez d'abord charger un fichier d'instances.")
            return
        
        print(f"Exécution de toutes les méthodes sur {len(self.problems)} instances...")
        results = self.experiment.run_all_methods_on_all_instances(self.problems)
        
        # Sauvegarder les résultats
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"results_all_{timestamp}.xlsx"
        summary_file = f"summary_all_{timestamp}.xlsx"
        
        self.experiment.save_results_to_excel(results, results_file)
        self.experiment.generate_summary_statistics(results, summary_file)
    
    def run_one_method_all_instances(self):
        """Exécute une méthode sur toutes les instances"""
        if not self.problems:
            print("Aucune instance disponible. Veuillez d'abord charger un fichier d'instances.")
            return
        
        method_name = self.select_method()
        if method_name is None:
            return
        
        print(f"Exécution de la méthode '{method_name}' sur {len(self.problems)} instances...")
        results = self.experiment.run_method_on_all_instances(method_name, self.problems)
        
        # Sauvegarder les résultats
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"results_{method_name.replace(' ', '_')}_{timestamp}.xlsx"
        
        self.experiment.save_results_to_excel(results, results_file)
    
    def compare_methods_one_instance(self):
        """Compare toutes les méthodes sur une instance spécifique"""
        problem = self.select_instance()
        if problem is None:
            return
        
        # Utiliser l'ID réel de l'instance plutôt que d'afficher toujours "instance 1"
        print(f"Comparaison de toutes les méthodes sur l'instance {problem.instance_id}...")
        results = self.experiment.run_all_methods_on_instance(problem)
        
        # Sauvegarder les résultats avec le bon ID d'instance
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"results_instance{problem.instance_id}_{timestamp}.xlsx"
        
        self.experiment.save_results_to_excel(results, results_file)
    
    def run_one_method_one_instance(self):
        """Exécute une méthode sur une instance spécifique"""
        problem = self.select_instance()
        if problem is None:
            return
        
        method_name = self.select_method()
        if method_name is None:
            return
        
        print(f"Exécution de la méthode '{method_name}' sur l'instance {problem.instance_id}...")
        result = self.experiment.run_method_on_instance(method_name, problem)
        
        # Afficher les résultats
        print("\nRésultats:")
        for key, value in result.items():
            print(f"{key}: {value}")
        
        # Sauvegarder les résultats
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"result_{method_name.replace(' ', '_')}_instance{problem.instance_id}_{timestamp}.xlsx"
        
        self.experiment.save_results_to_excel([result], results_file)
    
    def show_instance_details(self):
        """Affiche les détails d'une instance"""
        problem = self.select_instance()
        if problem is None:
            return
        
        details = self.experiment.get_instance_details(problem)
        
        print(f"\nDétails de l'instance {problem.instance_id}:")
        for key, value in details.items():
            if key in ["Processing Times", "Molds"]:
                print(f"{key}: ", end="")
                if len(value) > 10:
                    print(f"{value[:10]}... (+ {len(value) - 10} éléments)")
                else:
                    print(value)
            else:
                print(f"{key}: {value}")
    
    def run(self):
        """Exécute l'interface utilisateur"""
        print("Bienvenue dans le système d'expérimentation pour le problème des machines parallèles")
        
        # Charger les instances
        if not self.load_instances():
            print("Impossible de continuer sans instances. Veuillez réessayer.")
            return
        
        while True:
            self.display_menu()
            choice = input("\nEntrez votre choix (1-7): ")
            
            if choice == '1':
                self.run_all_methods_all_instances()
            elif choice == '2':
                self.run_one_method_all_instances()
            elif choice == '3':
                self.compare_methods_one_instance()
            elif choice == '4':
                self.run_one_method_one_instance()
            elif choice == '5':
                self.show_instance_details()
            elif choice == '6':
                self.load_instances()
            elif choice == '7':
                print("Merci d'avoir utilisé le système d'expérimentation. Au revoir!")
                break
            else:
                print("Choix invalide. Veuillez entrer un nombre entre 1 et 7.")
            
            input("\nAppuyez sur Entrée pour continuer...")


# Point d'entrée du programme
if __name__ == "__main__":
    # Configuration de matplotlib pour les graphiques
    plt.style.use('ggplot')
    
    # Lancer l'interface utilisateur
    cli = CLIInterface()
    cli.run()