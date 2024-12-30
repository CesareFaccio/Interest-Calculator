import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

class CompoundingInterest:

    def __init__(self, initValue, interest, interestUncertainty, interestVariance, interestRange, numInterests, monthlyAdditions, start, period):  # Constructor with a parameter
        self.initValue = initValue
        self.interest = interest
        self.interestUncertainty = interestUncertainty
        self.interestVariance = interestVariance
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

    def singleInterest(self):
        """
        calculates how value changes over time due to a single interest gain (also includes uncertainty in interest)
        """
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
                  f'\n final value = {result[-1]:.2f}')
        plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.show()

        return 0


    def singlePeriod(self):
        """
        calculates how the final values changes due to different interest rates
        """

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

    def manyRatesAndPeriods(self):
        """
        creates a heatmap showing final values for different interest rates and maturation periods
        """

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

    def priceSim(self,simNumber,plotTo):
        """
        this function runs a Brownian motion model simNumber amount of times,
        it then used the results to construct a PDF.
        This simulation takes into account monthly contributions.
        """

        finalValues = []
        plt.figure(figsize=(15, 6))
        t = np.linspace(0, (self.end) * 12, (self.end) * 12 * 30) / 12

        for j in range(simNumber):
            value = self.getStockPriceSimulation(self.initValue, self.interest, self.interestVariance, 1/12, steps=30)
            for i in range((self.end) * 12 - 1):
                value = np.append(value, self.getStockPriceSimulation(value[-1]+self.monthlyAdditions[i], self.interest, self.interestVariance, 1/12, steps=30))
            plt.plot(t, value)
            finalValues.append(value[-1])

        plt.title("Simulated Stock Prices (Geometric Brownian Motion)")
        plt.xlabel("Time (Years)")
        plt.ylabel("Stock Price")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        counts, bin_edges, patches = plt.hist(finalValues, bins=50, edgecolor='black')
        plt.xticks(bin_edges)
        plt.title(f'Estimated PDF \n{sum(self.monthlyAdditions) + self.initValue} contributed')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2
        expectationValue = np.sum(bin_values*counts/simNumber)
        plt.axvline(expectationValue, color='r', linestyle='--',label=f'Expected Value : {expectationValue:.2f}')

        for count, patch in zip(counts, patches):
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            if x < plotTo:
                plt.text(x, y, f'{(count/simNumber):.3f}', ha='center', va='bottom', fontsize=8)

        plt.xlim(0, plotTo)
        plt.legend()
        plt.show()

    def priceSimPDF(self,simNumber,plotTo):
        '''
        this function runs a Brownian motion model simNumber amount of times,
        it then used the results to construct a PDF.
        it then compares the results to the theoretical PDF.
        this simulation does not into account any monthly contributions.
        '''

        finalValues = []
        plt.figure(figsize=(10, 6))
        t = np.linspace(0, (self.end) * 12, (self.end) * 12 * 30) / 12

        for j in range(simNumber):
            value = self.getStockPriceSimulation(self.initValue, self.interest, self.interestVariance, 1/12, steps=30)
            for i in range((self.end) * 12 - 1):
                value = np.append(value,self.getStockPriceSimulation(value[-1], self.interest,self.interestVariance, 1 / 12, steps=30))
            plt.plot(t, value)
            finalValues.append(value[-1])

        plt.title("Simulated Stock Prices (Geometric Brownian Motion)")
        plt.xlabel("Time (Years)")
        plt.ylabel("Stock Price")
        plt.grid(True)
        plt.show()

        # Parameters for the GBM PDF
        S0 = self.initValue  # Initial stock price
        mu = self.interest  # Drift rate
        sigma = self.interestVariance  # Volatility
        t = self.end  # Time horizon

        x = np.linspace(0, plotTo, 1000)  # Price range
        pdf_values = [self.gbmPDF(price, S0, mu, sigma, t) for price in x]

        mean = S0 * np.exp(mu * t)  # Theoretical mean
        mode = S0 + np.exp((mu - 0.5 * sigma ** 2) * t)  # Theoretical mode

        plt.figure(figsize=(10, 6))
        plt.plot(x, pdf_values, label="GBM PDF", color="red")
        plt.hist(finalValues, bins=80,color = 'green' , edgecolor='green', density=True)
        plt.axvline(mean, color="blue", linestyle="--",linewidth=2,  label=f"Theoretical Mean: {mean:.2f}")
        plt.axvline(mode, color="brown", linestyle="--",linewidth=2, label=f"Theoretical Mode: {mode:.2f}")
        plt.title('Histogram of Data')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim(0, plotTo)
        plt.show()

    @staticmethod
    def gbmPDF(x, S0, mu, sigma, t):
        """
        calculates the PDF value for each x (price value)

        :param x: price value
        :param S0: Initial stock price
        :param mu: expected stock increase (yearly)
        :param sigma: expected stock volatility (yearly)
        :param t: period
        :return: PDF value
        """
        if x <= 0:
            return 0
        coeff = 1 / (x * np.sqrt(2 * np.pi * sigma ** 2 * t))
        exponent = -((np.log(x) - np.log(S0) - (mu - 0.5 * sigma ** 2) * t) ** 2) / (2 * sigma ** 2 * t)
        return coeff * np.exp(exponent)

    @staticmethod
    def plotStockPriceSimulation(initPrice, mu, sigma, T, steps):
        """
        Simulates and plots stock prices using Geometric Brownian Motion (GBM).

        Parameters:
        - initPrice: Initial stock price
        - mu: Expected return (drift)
        - sigma: Volatility (yearly)
        - T: Total time period (in years)
        - steps: Number of steps (typically trading days in a year)
        """

        dt = T / steps  # Time increment
        t = np.linspace(0, T, steps)  # Time array

        stepDifference = np.random.normal(0, np.sqrt(dt), steps)
        brownianMotion = np.cumsum(stepDifference)

        sim = initPrice * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * brownianMotion)

        # Plot the stock prices
        plt.figure(figsize=(10, 6))
        plt.plot(t, sim, lw=2)
        plt.title("Simulated Stock Prices (Geometric Brownian Motion)")
        plt.xlabel("Time (Years)")
        plt.ylabel("Stock Price")
        plt.grid(True)
        plt.show()

    @staticmethod
    def getStockPriceSimulation(initPrice, mu, sigma, T, steps):
        """
        Simulates and plots stock prices using Geometric Brownian Motion (GBM).

        Parameters:
        - initPrice: Initial stock price
        - mu: Expected return (drift)
        - sigma: Volatility (yearly)
        - T: Total time period (in years)
        - steps: Number of steps (typically trading days in a year)
        """

        dt = T / steps  # Time increment
        t = np.linspace(0, T, steps)  # Time array

        stepDifference = np.random.normal(0, np.sqrt(dt), steps)
        brownianMotion = np.cumsum(stepDifference)

        sim = initPrice * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * brownianMotion)

        return sim