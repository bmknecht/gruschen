
class ProgressPrinter:
    def __init__(self, max_iterations):
        self.max_iterations = int(max_iterations)
        self.iterations = 0
        self.last_printed_percent = -1

    def iterate_and_print(self):
        self.iterations += 1
        percent = int(round(self.iterations / self.max_iterations * 100))
        assert percent <= 100
        if percent > self.last_printed_percent:
            self.last_printed_percent = percent
            print('\r{} of {} ~ {} %'.format(self.iterations,
                                             self.max_iterations,
                                             percent),
                  end='')
        if self.iterations == self.max_iterations:
            # overwrite prev. message with whitespace
            print("\r... done" + (" "*30))
