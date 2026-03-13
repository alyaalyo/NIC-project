import random
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class LayerGene:
    """
    Representation of a single neural network layer.
    """
    def __init__(self, units, activation, dropout_rate):
        self.units = units          # number of neurons
        self.activation = activation  # activation function
        self.dropout_rate = dropout_rate  # dropout (after the layer)

    def __repr__(self):
        return f"Layer(units={self.units}, activation={self.activation}, dropout={self.dropout_rate})"


class ArchitectureGene:
    """
    Gene encoding a neural network architecture.
    Contains a list of layers (LayerGene) and possibly global parameters.
    """
    def __init__(self, layers):
        self.layers = layers  # list of LayerGene objects

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        return f"ArchitectureGene(layers={self.layers})"

    def to_dict(self):
        """Represent the gene as a dictionary (for serialization)."""
        return {
            'layers': [
                {'units': l.units, 'activation': l.activation, 'dropout_rate': l.dropout_rate}
                for l in self.layers
            ]
        }


def generate_random_individual(
    min_layers=1,
    max_layers=5,
    units_choices=[16, 32, 64, 128, 256],
    activation_choices=['relu', 'tanh', 'sigmoid'],
    dropout_range=(0.0, 0.5),
):
    """
    Generates a random individual (ArchitectureGene) with given constraints.
    
    Parameters:
        min_layers, max_layers: range for the number of layers.
        units_choices: list of allowed layer sizes.
        activation_choices: list of allowed activation functions.
        dropout_range: tuple (min, max) for the dropout rate.
        input_dim: ignored, but can be useful for compatibility checks.
    
    Returns:
        ArchitectureGene
    """
    num_layers = random.randint(min_layers, max_layers)
    layers = []
    for _ in range(num_layers):
        units = random.choice(units_choices)
        activation = random.choice(activation_choices)
        dropout_rate = round(random.uniform(*dropout_range), 2)  # round to two decimals
        layers.append(LayerGene(units, activation, dropout_rate))
    return ArchitectureGene(layers)


def build_mlp_from_gene(gene: ArchitectureGene, input_dim: int) -> models.Sequential:
    """
    Builds a Keras MLP model from an ArchitectureGene.
    
    Parameters:
        gene: ArchitectureGene object containing layer specifications.
        input_dim: number of input features.
    
    Returns:
        A compiled (but not yet trained) Sequential model.
    """
    model = models.Sequential()
    for i, layer_gene in enumerate(gene.layers):
        if i == 0:
            # First layer: specify input_shape
            model.add(layers.Dense(
                layer_gene.units,
                activation=layer_gene.activation,
                input_shape=(input_dim,)
            ))
        else:
            model.add(layers.Dense(
                layer_gene.units,
                activation=layer_gene.activation
            ))
        if layer_gene.dropout_rate > 0:
            model.add(layers.Dropout(layer_gene.dropout_rate))
    
    # Output layer for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Optionally compile (can be done later during training)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


def initialize_population(pop_size: int, input_dim: int, **kwargs) -> list:
    """
    Creates a population of random individuals and builds their MLP models.
    
    Parameters:
        pop_size: number of individuals in the population.
        input_dim: number of input features (used for model building).
        **kwargs: additional arguments passed to generate_random_individual.
    
    Returns:
        List of tuples (individual, model) for each member.
    """
    population = []
    for _ in range(pop_size):
        individual = generate_random_individual(**kwargs)
        model = build_mlp_from_gene(individual, input_dim)
        population.append((individual, model))
    return population

def mutate_individual(individual: ArchitectureGene, mutation_rate: float = 0.3, 
                      units_choices=[16, 32, 64, 128, 256],
                      activation_choices=['relu', 'tanh', 'sigmoid'],
                      dropout_range=(0.0, 0.5)) -> ArchitectureGene:
    """
    Mutates an individual by randomly changing layer parameters.
    Each layer has a probability mutation_rate to be mutated.
    """
    mutated_layers = []
    for layer in individual.layers:
        if random.random() < mutation_rate:
            # Randomly choose which parameter to mutate
            mutation_type = random.choice(['units', 'activation', 'dropout'])
            
            units = layer.units
            activation = layer.activation
            dropout_rate = layer.dropout_rate
            
            if mutation_type == 'units' or random.random() < 0.33:
                units = random.choice(units_choices)
            if mutation_type == 'activation' or random.random() < 0.33:
                activation = random.choice(activation_choices)
            if mutation_type == 'dropout' or random.random() < 0.33:
                dropout_rate = round(random.uniform(*dropout_range), 2)
                
            mutated_layers.append(LayerGene(units, activation, dropout_rate))
        else:
            mutated_layers.append(layer)
    
    return ArchitectureGene(mutated_layers)

def roulette_selection(population_with_fitness, num_parents: int):
    """
    Selects parents using roulette wheel selection based on fitness scores.
    population_with_fitness: list of tuples [(individual, model, fitness), ...]
    Returns list of selected parents (individuals only).
    """
    # Extract fitness values
    fitness_values = [fitness for _, _, fitness in population_with_fitness]
    
    # Handle negative fitness values by shifting to positive
    min_fitness = min(fitness_values)
    if min_fitness < 0:
        fitness_values = [f - min_fitness + 1e-10 for f in fitness_values]
    
    # Calculate total fitness and selection probabilities
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    
    # Select parents
    selected_indices = np.random.choice(
        len(population_with_fitness), 
        size=num_parents, 
        p=selection_probs, 
        replace=False
    )
    
    return [population_with_fitness[i][0] for i in selected_indices]


def single_point_crossover(parent1: ArchitectureGene, parent2: ArchitectureGene) -> tuple:
    """
    Performs single-point crossover between two parent architectures.
    Returns two child individuals.
    """
    # Find crossover point (can't be at the ends)
    min_len = min(len(parent1), len(parent2))
    if min_len < 2:
        # If too short, just return copies of parents
        return parent1, parent2
    
    crossover_point = random.randint(1, min_len - 1)
    
    # Create children by combining layers from both parents
    child1_layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
    child2_layers = parent2.layers[:crossover_point] + parent1.layers[crossover_point:]
    
    return ArchitectureGene(child1_layers), ArchitectureGene(child2_layers)

# Example usage
if __name__ == "__main__":
    # Example: input dimension (e.g., after feature extraction)
    INPUT_DIM = 100
    POP_SIZE = 5

    # Initialize population
    population = initialize_population(POP_SIZE, INPUT_DIM)

    # Display each individual and its model summary
    for idx, (ind, model) in enumerate(population):
        print(f"\n--- Individual {idx+1} ---")
        print("Gene:", ind)
        print("Model summary:")
        model.summary()
        print("Total parameters:", model.count_params())