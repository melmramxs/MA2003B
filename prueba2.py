import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import warnings

warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv('reto_db_final.csv')
df.fillna(df.mean(), inplace=True)

X = df.drop(['indice_lote', 'Friabilidad (%)', 'VM[kp]'], axis=1)
y1 = df['Friabilidad (%)']
y2 = df['VM[kp]']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled, y2, test_size=0.2, random_state=42)

model1 = LinearRegression().fit(X_train1, y_train1)
model2 = LinearRegression().fit(X_train2, y_train2)

# Define optimization problem
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Minimize friability, maximize hardness
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Get feature ranges for bounded initialization
ranges = list(zip(X.min().values, X.max().values))

def create_individual(ranges):
    return creator.Individual([np.random.uniform(low, high) for low, high in ranges])

def evaluate_individual(individual):
    scaled_individual = scaler.transform([individual])
    friability = model1.predict(scaled_individual)[0]
    hardness = model2.predict(scaled_individual)[0]
    return friability, hardness

# Custom crossover function
def custom_cx(ind1, ind2):
    size = len(ind1)
    cxpoint1, cxpoint2 = sorted(np.random.choice(range(size), 2, replace=False))
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2

# Custom mutation function
def custom_mut(individual, indpb, mu, sigma):
    size = len(individual)
    for i in range(size):
        if np.random.random() < indpb:
            individual[i] += np.random.normal(mu, sigma)
            individual[i] = np.clip(individual[i], ranges[i][0], ranges[i][1])
    return individual,

# Set up the evolutionary algorithm
toolbox = base.Toolbox()
toolbox.register("individual", create_individual, ranges=ranges)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", custom_cx)
toolbox.register("mutate", custom_mut, indpb=0.2, mu=0, sigma=0.5)
toolbox.register("select", tools.selNSGA2)

# Algorithm parameters
NGEN = 100
POPSIZE = 200
CXPB = 0.9
MUTPB = 0.1

# Initialize population and run the algorithm
population = toolbox.population(n=POPSIZE)
algorithms.eaMuPlusLambda(population, toolbox, mu=POPSIZE, lambda_=POPSIZE, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                          stats=None, halloffame=None, verbose=True)

# Get Pareto front
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

# Display results
print("\nPareto Front Solutions:")
for idx, individual in enumerate(pareto_front, 1):
    friability, hardness = evaluate_individual(individual)
    print(f"Solution {idx}:")
    for name, value in zip(X.columns, individual):
        print(f"  {name}: {value:.4f}")
    print(f"  Friability: {friability:.4f}")
    print(f"  Hardness: {hardness:.4f}\n")