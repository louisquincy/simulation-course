import random
import statistics

N = 100000


class MultiKongGen:
    def __init__(self):
        self.M = 2 ** 63
        self.beta = 2 ** 32 + 3
        self.x0 = self.beta
    def anabios(self):
        self.x0 = (self.beta * self.x0) % self.M
        return self.x0 / self.M

def skill(gen, n=N):
    return [gen() for _ in range(n)]

def wisdom(numbers):
    mean = statistics.mean(numbers)
    dispersion = statistics.variance(numbers)
    return mean, dispersion

def main():
    gen = MultiKongGen()
    numbers = skill(gen.anabios, N)
    mean, dispersion = wisdom(numbers)

    random.seed(10032005)
    py_numbers = skill(random.random, N)
    py_mean, py_var = wisdom(py_numbers)

    theory_mean = 0.5
    theory_var = 1.0 / 12.0

    print(f"{'Источник':<20} {'Среднее':<15} {'Дисперсия':<15}")
    print("-" * 50)
    print(f"{'Теория':<20} {theory_mean:<15.6f} {theory_var:<15.6f}")
    print(f"{'МКГ (свой)':<20} {mean:<15.6f} {dispersion:<15.6f}")
    print(f"{'random() Python':<20} {py_mean:<15.6f} {py_var:<15.6f}")
    print()
    print(f"Δ среднее МКГ:      {abs(mean - theory_mean):.6e}")
    print(f"Δ среднее random:   {abs(py_mean - theory_mean):.6e}")
    print(f"Δ дисперсия МКГ:    {abs(dispersion - theory_var):.6e}")
    print(f"Δ дисперсия random: {abs(py_var - theory_var):.6e}")

if __name__ == "__main__":
    main()