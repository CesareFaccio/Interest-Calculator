import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

class compoundingInterest:

    def __init__(self, initValue, interest, interestUncertainty, interestRange, numInterests, monthlyAdditions, start, period):  # Constructor with a parameter
        self.initValue = initValue
        self.interest = interest
        self.interestUncertainty = interestUncertainty
        self.interestRange = interestRange
        self.numInterests = numInterests
        self.monthlyAdditions = monthlyAdditions
        self.start = start
        self.period = period
        self.end = start + period
        self.interests = np.linspace(self.interest-self.interestRange, self.interest+self.interestRange, self.numInterests)
        self.periods = np.arange(self.start , self.end + 1)

    def compoundInterest(self, startValue, rate, monthlyAdditions, time):
        result = startValue
        multiplier = 1 + rate/12
        for i in range(time*12):
            result = (result + monthlyAdditions[i]) * multiplier
        roundedFinal = round(result, 2)
        return roundedFinal

    #calculates how value changes over time due to a single interest gain (also includes uncertainty in interest)
    def singleInterest(self):
        result = []
        resultLower = []
        resultUpper = []
        noInterest = []
        times = []

        for time in range(self.start, self.end + 1):
            times.append(time)
            result.append(self.compoundInterest(self.initValue, self.interest, self.monthlyAdditions, time))
            resultLower.append(self.compoundInterest(self.initValue, self.interest-self.interestUncertainty,self.monthlyAdditions, time))
            resultUpper.append(self.compoundInterest(self.initValue, self.interest+ self.interestUncertainty,self.monthlyAdditions, time))
            noInterest.append(self.compoundInterest( self.initValue, 0,self.monthlyAdditions, time))

        plt.figure(figsize=(8, 7))
        plt.plot(times, result, marker='o', color='blue', linestyle='-', label=f'{self.interest}')
        plt.plot(times, resultLower, marker='o', color='red', linestyle='-', label=f'{(self.interest-self.interestUncertainty):.3f}')
        plt.plot(times, resultUpper, marker='o', color='green', linestyle='-', label=f'{(self.interest+self.interestUncertainty):.3f}')
        plt.plot(times, noInterest, marker='o', color='brown', linestyle='-', label=f'no interest')

        plt.xlabel('Time (years)')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.title(f'results over {self.end} years starting with {self.initValue}'
                  f'\n including mean monthly additions of {sum(self.monthlyAdditions)/len(self.monthlyAdditions)}, {noInterest[-1]} contributed '
                  f'\n final value = {result[-1]}')
        plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.show()

        return 0

    #calculates how the final values changes due to different interest rates
    def singlePeriod(self):

        results = []
        lowerResults = []
        upperResults = []
        for rate in self.interests:
            results.append(self.compoundInterest(self.initValue, rate, self.monthlyAdditions, self.end))
            lowerResults.append(self.compoundInterest(self.initValue, rate - self.interestUncertainty,self.monthlyAdditions, self.end))
            upperResults.append(self.compoundInterest(self.initValue, rate + self.interestUncertainty,self.monthlyAdditions , self.end))

        plt.figure(figsize=(8, 7))
        plt.plot(self.interests, results, marker='o', color='blue', linestyle='-', label='base interest')
        plt.plot(self.interests, lowerResults, marker='o', color='red', linestyle='-', label='base interest minus uncertainty')
        plt.plot(self.interests, upperResults, marker='o', color='green', linestyle='-', label='base interest plus uncertainty')
        plt.hlines(y=sum(self.monthlyAdditions) + self.initValue, xmin=self.interests[0], xmax=self.interests[-1], color='brown', linestyle='--', label='no interest')

        plt.xlabel('Interest Rate (decimal)')
        plt.ylabel('Final Value')
        plt.title(f'results over {self.end} years starting with {self.initValue} '
                  f'\n including mean monthly additions of {sum(self.monthlyAdditions)/len(self.monthlyAdditions)}, {sum(self.monthlyAdditions) + self.initValue} contributed')
        plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.show()

        return 0

    #creates a heatmap showing final values for different interest rates and maturation periods
    def manyRatesAndPeriods(self):

        rate_edges = np.linspace(self.interest-self.interestRange, self.interest+self.interestRange, self.numInterests + 1)
        period_edges = np.arange(self.start , (self.end) + 2)

        final_values = np.array([
            [self.compoundInterest(self.initValue, rate, self.monthlyAdditions, time)
            for rate in self.interests]
            for time in self.periods
        ])

        norm = Normalize(vmin=final_values.min(), vmax=final_values.max())

        cmap_name = 'custom_gradient'
        colors = [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

        plt.figure(figsize=(19, 9))  # Adjusted figure size
        plt.pcolormesh(rate_edges, period_edges, final_values, shading="flat", cmap="RdBu", edgecolors='black',linewidth=0.5)
        plt.colorbar(label="Final Value")
        plt.title(f"Final Value Heatmap ({self.initValue} initial investment, including mean monthly additions of {sum(self.monthlyAdditions)/len(self.monthlyAdditions)}, {sum(self.monthlyAdditions) + self.initValue} contributed)")
        plt.xlabel("Interest Rate (decimal)")
        plt.ylabel("Time Elapsed (years)")

        plt.xticks(rate_edges[:-1] + np.diff(rate_edges) / 2, [f"{rate:.4f}" for rate in self.interests])
        plt.yticks(period_edges[:-1] + 0.5, self.periods)

        for i, period in enumerate(self.periods):
            for j, rate in enumerate(self.interests):
                value = final_values[i, j]
                color = cmap(norm(value))
                plt.text(
                    rate_edges[j] + np.diff(rate_edges)[0] / 2,
                    period_edges[i] + 0.5,
                    f"{final_values[i, j]:.2f}",
                    color=color,
                    ha="center", va="center", fontsize=8, weight="bold"
                )

        plt.show()

        return 0