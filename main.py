import random
import itertools


board_count = 8

board_count -= 1


def generate_first_population(pop_size):
    return [[
        random.randint(0, board_count) for _ in range(0, board_count + 1)
    ] for _ in range(0, pop_size + 1)]


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


def cross(first_chromosome, second_chromosome):

    max_number = board_count

    shift = random.randint(0, max_number.bit_length() - 1)
    first_mask = max_number >> shift

    second_mask = first_mask ^ max_number

    first_child_chromosome = first_chromosome & first_mask ^ second_chromosome & second_mask
    second_child_chromosome = second_chromosome & first_mask ^ first_chromosome & second_mask

    return first_child_chromosome, second_child_chromosome


def cross_population(population, prop):
    result = []
    for first_individ, second_individ in list(zip(*[iter(population)] * 2)):
        if prop >= random.uniform(0, 1):
            first_child = []
            second_child = []
            for index in range(0, len(first_individ)):
                first_child_chromosome, second_child_chromosome = cross(first_individ[index], second_individ[index])
                first_child.append(first_child_chromosome)
                second_child.append(second_child_chromosome)
            result.extend((first_child, second_child))
    return result


def mutate(population, prop):
    for individ in population:
        if prop >= random.uniform(0, 1):
            index_gen_for_mutate = random.randint(0, len(individ) - 1)
            individ[index_gen_for_mutate] = random.randint(0, 7)

    return population


def fitness_function(individ):
    combination_weight = 100 / len(list(itertools.combinations(range(0, board_count + 1), 2)))
    result = 0
    coordinates_individ = list(enumerate(individ))

    for current_x, current_y in coordinates_individ:
        for checked_x, checked_y in coordinates_individ[current_x + 1:]:
            x_diff = checked_x - current_x
            y_diff = checked_y - current_y
            if y_diff == 0 or (current_y + x_diff == checked_y) or (current_y - x_diff == checked_y):
                result += 1
    return 100 - result * combination_weight


def selection(population, fitnesses, num):
    total_weight = sum(fitnesses)
    fitnesses_ref = [f / total_weight for f in fitnesses]
    probs = [sum(fitnesses_ref[:index + 1]) for index, _ in enumerate(fitnesses_ref)]

    result = []

    for _ in range(0, num):
        rand = random.uniform(0, 1)
        for i, individual in enumerate(population):
            if rand <= probs[i]:
                result.append(individual)
                break
    return result


class Solver_8_queens:
    def __init__(self, pop_size=1000, cross_prob=0.30, mut_prob=0.25):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob

    def solve(self, min_fitness=1, max_epochs=1000):
        best_fit = 0
        best_individual = None
        epoch_num = None
        is_done = False

        population = generate_first_population(self.pop_size)
        for epoch_num in range(0, max_epochs):
            fitnesses = []
            for individual in population:
                individual_ff = fitness_function(individual)

                if individual_ff / 100 >= min_fitness:
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

            population = selection(population, fitnesses, self.pop_size)
            population = cross_population(population, self.cross_prob)
            population = mutate(population, self.mut_prob)
        return best_fit, epoch_num, visualization(best_individual)


if __name__ == "__main__":
    solver = Solver_8_queens()
    best_fit, epoch_num, visual = solver.solve()
    print(str.format("best_fit : {0},\nepoch_num: {1},\nvisual:\n{2}", best_fit, epoch_num, visual))



