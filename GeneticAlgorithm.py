import random
import copy
import cv2

class Settings(object):
    imageName = input('Write the image name: ')
    def __init__(self):
        self.NO_OF_GENES = 400
        self.MUTATION_PROBABILITY = 0.002
        self.POPULATION_SIZE = 600
        self.IDEAL_FITNESS = 0
        self.ELITE_CARRY_OVER = 20
        self.FITNESS_CATEGORY = "minimize"
        self.CROSSOVER_PROBABILITY = 0.95
        self.GENE_TYPE = "binary"

class Chromosome(object):
    def __init__(self, config):
        self.config = config
        self.genes = [random.randint(0, 1) for i in range(0, self.config.NO_OF_GENES)]
        self.fitness = 0
        self.originalFitness = 0.0
        self.normalizedFitness = 0.0
        self.endRange = 0.0
    
    def mutate(self):
        for i in range(0, len(self.genes)):
            if random.uniform(0, 1) <= self.config.MUTATION_PROBABILITY:
                self.genes[i] = self.__flipGene__(self.genes[i])
    
    def __flipGene__(self, geneValue):
        if geneValue == 0:
            return 1
        return 0


class RW(object):
    def __init__(self, chromosomes, config):
        self.chromosomes = chromosomes
        self.config = config
        self.__createCumulativeProbabilities__()
    
    def RWSelection(self):
        # Select first chromosome
        random1 = random.uniform(0, 1)
        chromosome1 = None
        for chromosome in self.chromosomes:
            if random1 <= chromosome.endRange:
                chromosome1 = Chromosome(self.config)
                chromosome1.genes = chromosome.genes
                break
        
        # Select second chromosome
        chromosome2 = None
        while True:
            random2 = random.uniform(0, 1)
            for chromosome in self.chromosomes:
                if random2 <= chromosome.endRange:
                    chromosome2 = Chromosome(self.config)
                    chromosome2.genes = chromosome.genes
                    
                    if chromosome1.genes != chromosome2.genes:
                        return chromosome1, chromosome2

    def __createCumulativeProbabilities__(self):
        self.__calculateCumulativeSum__()
        self.__getNomalizedFitness__()
        currentSum = 0
        for chromosome in self.chromosomes:
            currentSum += chromosome.normalizedFitness
            chromosome.endRange = currentSum
    
    def __calculateCumulativeSum__(self):
        cumSum = 0
        for chromosome in self.chromosomes:
            cumSum += chromosome.fitness
        self.cumSum = cumSum
    
    def __getNomalizedFitness__(self):
        for chromosome in self.chromosomes:
            chromosome.normalizedFitness = chromosome.fitness / self.cumSum

class mainAlgorithm(object):
    def __init__(self):
        self.config = Settings()
        self.GENERATION_COUNT = 0
        self.selectedChromosomes = {}
    
    def __crossover__(self, chromosome1, chromosome2):
        # Randomly create a crossover point
        crossoverPoint = random.randint(1, len(chromosome1.genes) - 1)
        # Prepare child genes
        child1Genes = chromosome1.genes[0 : crossoverPoint] + chromosome2.genes[crossoverPoint : ]
        child2Genes = chromosome2.genes[0 : crossoverPoint] + chromosome1.genes[crossoverPoint : ]
        chromosomeToConsider1 = Chromosome(self.config)
        chromosomeToConsider2 = Chromosome(self.config)

        if random.uniform(0, 1) <= self.config.CROSSOVER_PROBABILITY:
            chromosomeToConsider1.genes = child1Genes
        else:
            chromosomeToConsider1.genes = chromosome1.genes.copy()
        
        if random.uniform(0, 1) <= self.config.CROSSOVER_PROBABILITY:
            chromosomeToConsider2.genes = child2Genes
        else:
            chromosomeToConsider2.genes = chromosome2.genes.copy()

        return chromosomeToConsider1, chromosomeToConsider2
    
    def __originShiftIfNegativeFitnesses__(self, chromosomes):
        # Find minimum fitness
        minimumFitness = min(chromosome.fitness for chromosome in chromosomes)

        # Origin shift only if fitnesses are negative
        # Here we are trying to make all fitnesses as positive values for easier execution.
        if minimumFitness >= 0:
            return
        
        for chromosome in chromosomes:
            chromosome.fitness += minimumFitness * -1
    
    def simulateEvolution(self, noOfGeneration, fitnessFunction, stepExecution = None):
        chromosomes = [Chromosome(self.config) for i in range(0, self.config.POPULATION_SIZE)]
        bestIndividual = Chromosome(self.config)
        bestIndividual.fitness = -1

        for generation in range(0, noOfGeneration):
            print(">> Generation = [" + str(self.GENERATION_COUNT)+"]")
            
            # Map of already created chromosomes
            # Helps to avoid including duplicate chromosomes which might mess with the algorithm.
            self.selectedChromosomes = {} 

            # Calculate Fitnesses
            for chromosome in chromosomes:
                fitnessValue = fitnessFunction(chromosome)
                chromosome.originalFitness = fitnessValue

                if self.config.FITNESS_CATEGORY == 'minimize':

                    # Reverse fitness in case of reverse category
                    fitnessValue *= -1
                chromosome.fitness = fitnessValue
            
            # Origin shift chromosomes in case of negative fitness values
            self.__originShiftIfNegativeFitnesses__(chromosomes)

            # Create Roulette Wheel for Current Generation
            rW = RW(chromosomes, self.config)
            nextGenChromosomes = []

            # Create next Generation by selection
            for i in range(0, int(self.config.POPULATION_SIZE / 2)):
                # Select two chromosomes from roulette wheel
                chromosome1, chromosome2 = rW.RWSelection()

                # Cross Over the chromosomes
                chromosome1, chromosome2 = self.__crossover__(chromosome1, chromosome2)
                
                # Mutate the new chromosomes
                chromosome1.mutate()
                chromosome2.mutate()

                # Add them to the New Generation Pool if genes are unique
                if (str(chromosome1.genes) not in self.selectedChromosomes):
                    nextGenChromosomes.append(chromosome1)
                    self.selectedChromosomes[str(chromosome1.genes)] = 1

                if (str(chromosome2.genes) not in self.selectedChromosomes):
                    nextGenChromosomes.append(chromosome2)
                    self.selectedChromosomes[str(chromosome2.genes)] = 1
                        
            # Sort Chromosomes based on fitness values
            chromosomes.sort(key = lambda x : x.fitness, reverse = True)
            
            # Save best individual
            bestIndividual = copy.deepcopy(chromosomes[0])
            
            # Check if Ideal fitness has been reached
            # If so, then return
            if bestIndividual.originalFitness == self.config.IDEAL_FITNESS:
                return bestIndividual
            
            # Carry over Elites to next Generation if they donot already exist
            for i in range(0, self.config.ELITE_CARRY_OVER):
                if (str(chromosomes[i].genes) not in self.selectedChromosomes):
                    nextGenChromosomes[len(nextGenChromosomes) - i - 1] = copy.deepcopy(chromosomes[i])
            
            # Execute Stepper
            if stepExecution != None:
                stepExecution(generationNumber = self.GENERATION_COUNT, bestIndividual = chromosomes[0])

            self.GENERATION_COUNT += 1
            print("Best => " + str(chromosomes[0].originalFitness) +"")
            chromosomes = nextGenChromosomes
        return bestIndividual
