from compoundingInterest import compoundingInterest

def main():
    initValue = 500
    interest = 0.04
    interestUncertainty = 0.02
    interestRange = 0.02
    numInterests = 5

    monthlyAddition = 20
    monthlyAdditions = [monthlyAddition] * (14 * 12)

    start = 10
    period = 4

    financeCalc = compoundingInterest(initValue,interest, interestUncertainty, interestRange, numInterests, monthlyAdditions, start, period)
    financeCalc.singleInterest()
    financeCalc.singlePeriod()
    financeCalc.manyRatesAndPeriods()

if __name__ == "__main__":
    main()