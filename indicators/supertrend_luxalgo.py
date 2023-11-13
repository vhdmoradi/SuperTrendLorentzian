class Supertrend:
    def __init__(self, upper, lower, output, perf, factor, trend):
        self.upper = upper
        self.lower = lower
        self.output = output
        self.perf = perf
        self.factor = factor
        self.trend = trend


# Supertrend instances stored in a list
holder = []

# List of factors
factors = []
