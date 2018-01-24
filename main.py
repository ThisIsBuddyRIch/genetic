import random
import itertools
from enum import Enum


class SelectionMethod(Enum):
    WHEEL = 0
    TOURNAMENT = 1


class CrossoverMethod(Enum):
    SINGLEPOINT = 0
    MULTIPOINT = 1


class PointGenerator:
    def __init__(self, crossover_method, max_point):
        self.crossover_method = crossover_method
        self.max_point = max_point
        self.single_point = None

    def get_point(self):
        if self.crossover_method == CrossoverMethod.SINGLEPOINT:
            if self.single_point is None:
                self.single_point = random.randint(0, self.max_point)
            return self.single_point
        elif self.crossover_method == CrossoverMethod.MULTIPOINT:
            return random.randint(0, self.max_point)
        else:
            raise RuntimeError(str.format('Unknown Crossover Method : {0}', self.crossover_method))


class Solver_8_queens:
    board_count = 7

    def __init__(self, pop_size=1000, cross_prob=0.30, mut_prob=0.25, crossover_method=CrossoverMethod.SINGLEPOINT, selection_method = SelectionMethod.WHEEL):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.crossover_method = crossover_method
        self.selection_method = selection_method

    def solve(self, min_fitness=1, max_epochs=1000):
        best_fit = 0
        best_individual = None
        epoch_num = 0
        is_done = False

        if min_fitness is None and max_epochs is None:
            raise RuntimeError("Min_fitness and max_epochs is none at the same time")

        population = self.generate_first_population()
        while max_epochs is None or epoch_num < max_epochs:
            fitnesses = []
            for individual in population:
                individual_ff = self.fitness_function(individual)

                if min_fitness is not None and individual_ff >= min_fitness:
                    best_fit = individual_ff
                    best_individual = individual
                    is_done = True
                    break
                else:
                    fitnesses.append(individual_ff)
                    if best_fit < individual_ff:
                        best_fit = individual_ff
                        best_individual = individual
            if is_done:
                break
            if self.selection_method == SelectionMethod.WHEEL:
                population = self.wheel_selection(population, fitnesses, self.pop_size)
            elif self.selection_method == SelectionMethod.TOURNAMENT:
                population = self.tournament_selection(population, fitnesses, self.pop_size)
            else:
                RuntimeError(str.format('Unknown Selection Method : {0}', self.selection_method))

            population = self.crossover_population(population, self.cross_prob)
            population = self.mutate(population, self.mut_prob)
            epoch_num += 1
        return best_fit, epoch_num, self.visualization(best_individual)

    def generate_first_population(self):
        return [[
            random.randint(0, Solver_8_queens.board_count) for _ in range(0, Solver_8_queens.board_count + 1)
        ] for _ in range(0, self.pop_size)]

    @staticmethod
    def visualization(individual):
        """
        Function for visualizing answer
        :param individual: vector when index - x in chess board and item - y in chess board
        :return: string for visualization
        """

        view_list = list(individual)

        tpl_len = len(view_list)
        max_tpl = max(view_list)
        if max_tpl > tpl_len:
            raise RuntimeError("tpl_len > max_tpl, tpl_len : {0} max_tpl : {1}".format(tpl_len, max_tpl))

        template = "+" * len(view_list) + "\n"
        result = [template] * tpl_len
        for item in view_list:
            visual_str = list(result[tpl_len - item - 1])
            i = view_list.index(item)
            visual_str[i] = "Q"
            view_list[i] = -1
            result[tpl_len - item - 1] = "".join(visual_str)

        return "".join(result)

    @staticmethod
    def crossover(first_chromosome, second_chromosome, point_gen):

        first_mask = Solver_8_queens.board_count >> point_gen.get_point()
        second_mask = first_mask ^ Solver_8_queens.board_count

        first_child_chromosome = first_chromosome & first_mask ^ second_chromosome & second_mask
        second_child_chromosome = second_chromosome & first_mask ^ first_chromosome & second_mask

        return first_child_chromosome, second_child_chromosome

    def crossover_population(self, population, prop):
        result = []
        for first_individ, second_individ in self.parents_choiсe(population):
            if prop >= random.uniform(0, 1):
                first_child = []
                second_child = []
                point_gen = PointGenerator(self.crossover_method,
                                           Solver_8_queens.board_count.bit_length() - 1)
                for index in range(0, len(first_individ)):
                    first_child_chromosome, second_child_chromosome = self.crossover(first_individ[index],
                                                                                     second_individ[index], point_gen)
                    first_child.append(first_child_chromosome)
                    second_child.append(second_child_chromosome)
                result.extend((first_child, second_child))
        return result

    @staticmethod
    def mutate(population, prop):
        for individ in population:
            if prop >= random.uniform(0, 1):
                index_gen_for_mutate = random.randint(0, len(individ) - 1)
                individ[index_gen_for_mutate] = random.randint(0, 7)

        return population

    def parents_choiсe(self, population):
        for _ in range(0, len(population)):
            yield self._get_random_population_item(population), self._get_random_population_item(population)

    @staticmethod
    def fitness_function(individual):
        combination_weight = 1 / len(list(itertools.combinations(range(0, Solver_8_queens.board_count + 1), 2)))
        result = 0
        coordinates_individual = list(enumerate(individual))

        for current_x, current_y in coordinates_individual:
            for checked_x, checked_y in coordinates_individual[current_x + 1:]:
                x_diff = checked_x - current_x
                y_diff = checked_y - current_y
                if y_diff == 0 or (current_y + x_diff == checked_y) or (current_y - x_diff == checked_y):
                    result += 1
        return 1 - result * combination_weight

    @staticmethod
    def _get_random_population_item(population):
        return population[random.randint(0, len(population) - 1)]

    @staticmethod
    def wheel_selection(population, fitnesses, num):
        total_weight = sum(fitnesses)
        fitnesses_refs = [f / total_weight for f in fitnesses]

        probs = [fitnesses_refs[0]]
        for ref in fitnesses_refs[1:]:
            probs.append(probs[-1] + ref)
        result = []

        for _ in range(0, num):
            rand = random.uniform(0, 1)
            for i, individual in enumerate(population):
                if rand <= probs[i]:
                    result.append(individual)
                    break
        return result

    @staticmethod
    def tournament_selection(population, fitnesses, num):
        result = []
        for _ in range(0, num):
            first_contender_index = random.randint(0, len(population) - 1)
            second_contender_index = random.randint(0, len(population) - 1)

            if fitnesses[first_contender_index] > fitnesses[second_contender_index]:
                result.append(population[first_contender_index])
            else:
                result.append(population[second_contender_index])
        return result


if __name__ == "__main__":
    solver = Solver_8_queens(selection_method=SelectionMethod.TOURNAMENT)
    best_fit, epoch_num, visual = solver.solve(max_epochs=None)
    print(str.format("best_fit : {0}\nepoch_num: {1}\nvisual:\n{2}", best_fit, epoch_num, visual))
