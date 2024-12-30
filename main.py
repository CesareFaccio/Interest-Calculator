from compoundingInterest import CompoundingInterest
from loanCalculator import LoanCalculator
import numpy as np

def main():

    initValue = 2000
    interest = 0.0588
    interestUncertainty = 0.02 #only effects plots
    interestVariance = 0.1938 #only effects simulation
    interestRange = 0.02
    numInterests = 5

    start = 0
    period = 15

    monthlyAddition = 20
    monthlyAdditions = [monthlyAddition] * ((start+period) * 12)

    financeCalc = CompoundingInterest(initValue,interest, interestUncertainty, interestVariance, interestRange, numInterests, monthlyAdditions, start, period)
    financeCalc.singleInterest()
    financeCalc.singlePeriod()
    financeCalc.manyRatesAndPeriods()
    financeCalc.priceSim(10000,25000)
    financeCalc.priceSimPDF(1000,15000)

    loanAmount = 100000
    interestRate = 0.09
    interestRateRange = np.linspace(0.02, 0.04, 10)
    monthlyPayments = 800
    monthlyPaymentRange = np.linspace(750.1, 800, 50)

    loanCalculator = LoanCalculator(loanAmount, interestRate,interestRateRange, monthlyPayments, monthlyPaymentRange)
    loanCalculator.plotLoan()
    loanCalculator.loanTimesPayments()
    loanCalculator.loanTimesRates()

if __name__ == "__main__":
    main()