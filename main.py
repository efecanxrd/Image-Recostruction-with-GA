import cv2
import numpy as np
from GeneticAlgorithm import mainAlgorithm
from GeneticAlgorithm import Settings

def fitness(chromosome):
    fitness = 0
    for i in range(0, len(imageFlattened)):
        fitness += pow((imageFlattened[i] - chromosome.genes[i]), 2)
    return fitness    

image = cv2.imread(Settings.imageName, 0)
imageFlattened = image.flatten().tolist()

# Convert 255's to 1 and rest to 0
# This is done in order to create binary image
for i in range(0, len(imageFlattened)):
    if imageFlattened[i] == 255:
        imageFlattened[i] = 1
    else:
        imageFlattened[i] = 0

image = np.reshape(imageFlattened, (20, 20)) * 255.0

# Step Executor Function
# Called after Each Generation for Any Action That the Programmer might want to execute
# Here we use this method to show the formed image at each step
def stepExecutor(generationNumber, bestIndividual):
    bestIndividual = np.array(bestIndividual.genes)
    bestIndividual = bestIndividual * 255.0
    bestIndividual = np.reshape(bestIndividual, (20, 20))
    bestIndividual = cv2.resize(bestIndividual, (200, 200))

    # Show original Image and Formed Image at each step
    cv2.imshow("ORIGINAL", cv2.resize(image, (200, 200)))
    cv2.imshow("FORMED", bestIndividual)
    key = cv2.waitKey(1)

util = mainAlgorithm()
node = util.simulateEvolution(300, fitness, stepExecution = stepExecutor)
npde = np.array(node.genes)
node = node * 255.0
node = np.reshape(node, (20, 20))
cv2.imshow("Formed", node)
cv2.waitKey(0)
