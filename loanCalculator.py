from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

class LoanCalculator:

    def __init__(self, loanAmount, interestRate, interestRateRange, monthlyPayment, monthlyPaymentRange,):
        self.loanAmount = loanAmount
        self.interestRate = interestRate
        self.monthlyRate = interestRate / 12
        self.interestRateRange = interestRateRange
        self.monthlyInterestRateRange = interestRateRange / 12
        self.monthlyPayment = monthlyPayment
        self.monthlyPaymentRange = monthlyPaymentRange

    def plotLoan(self):
        """
        plots how loan amount (amount left to pay off) changes over time
        """
        value = [self.loanAmount]
        for i in range(100*12):
            value.append(((1+self.monthlyRate) * value[-1]) -self.monthlyPayment)
            if value[-1] < 0:
                break

        t = np.linspace(0, len(value), len(value))/12

        plt.figure(figsize=(8, 7))
        plt.plot(t,value)
        plt.xlabel('Time (years)')
        plt.ylabel('Loan Value Remaining')
        if value[-1] < 0:
            plt.title(f'Loan Value Remaining vs Time, time to pay-off ({t[-1]:.2f} years) '
                      f'\n loan amount {self.loanAmount}, loan interest rate {self.interestRate} ')
        else:
            plt.title(f'Loan Value Remaining vs Time \n'
                      f'loan cannot effectively be paid off, value after 100 years ({value[-1]:.2f})')
        plt.show()

    def loanTimesPayments(self):
        """
        shows how time to pay off loan depends on monthly payment amount
        """
        minValue = self.loanAmount*self.monthlyRate
        range = [item for item in self.monthlyPaymentRange if item > minValue]
        if len(range) != len(self.monthlyPaymentRange):
            print('Values that would never lead to loan being payed off where included in range '
                  '\nrange has been adjusted')
        times = []
        for mpr in range:
            time = calculatePayoffTime(self.loanAmount,self.interestRate, mpr)
            times.append(time)

        plt.figure(figsize=(8, 7))
        plt.plot(range,times, marker='o', color='blue', linestyle='-')
        plt.xlabel('Monthly Repayment Amount')
        plt.ylabel('Time to Repay Loan (years)')
        plt.title('Monthly Repayment Amount vs Time to Repay Loan \n'
                  f'loan rate {self.interestRate}, loan amount {self.loanAmount}')
        plt.show()

    def loanTimesRates(self):
        """
        shows how time to pay off loan depends on loan interest rate
        """
        if self.monthlyPayment <= self.loanAmount*self.interestRate/12:
            print('monthly payment too low, default to lowest value')
            monthlyRepay = self.loanAmount*self.interestRate/12 + 0.1
        else:
            monthlyRepay = self.monthlyPayment
        times = []
        for interest in self.interestRateRange:
            time = calculatePayoffTime(self.loanAmount,interest, monthlyRepay)
            times.append(time)

        plt.figure(figsize=(8, 7))
        plt.plot(self.interestRateRange,times, marker='o', color='blue', linestyle='-')
        plt.xlabel('Loan Interest Rate')
        plt.ylabel('Time to Repay Loan (years)')
        plt.title(f'Loan Interest Rate vs Time to Repay Loan \n'
                  f'(monthly Payment of {monthlyRepay}) (loan amount {self.loanAmount})')
        plt.show()

@staticmethod
def calculatePayoffTime(loanAmount, annualInterestRate, monthlyPayment):
    monthlyInterestRate = annualInterestRate / 12
    numerator = np.log(monthlyPayment / (monthlyPayment - loanAmount * monthlyInterestRate))
    denominator = np.log(1 + monthlyInterestRate)
    months_to_payoff = numerator / denominator
    years = months_to_payoff / 12
    return years