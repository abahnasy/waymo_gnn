import time
def timeit(f):
    def timed(*args, **kw):
        tick = time.time()
        res = f(*args, **kw)
        tock = time.time()
        class_name = type(args[0]).__name__
        print("{} {} function time is: {:.5f}".format(class_name, f.__name__, tock - tick)) # ex. Preprocess call funtion time: xxx
        return res
    return timed

from contextlib import contextmanager
from abc import ABC, abstractmethod
import cProfile
import pstats
import io
import logging

logger = logging.getLogger(__name__)

class BaseProfiler(ABC):
    """
    If you wish to write a custom profiler, you should inhereit from this class.
    """

    @abstractmethod
    def start(self, action_name):
        """
        Defines how to start recording an action.
        """
        pass

    @abstractmethod
    def stop(self, action_name):
        """
        Defines how to record the duration once an action is complete.
        """
        pass

    @contextmanager
    def profile(self, action_name):
        """
        Yields a context manager to encapsulate the scope of a profiled action.
        Example::
            with self.profile('load training data'):
                # load training data code
        The profiler will start once you've entered the context and will automatically
        stop once you exit the code block.
        """
        try:
            self.start(action_name)
            yield action_name
        finally:
            self.stop(action_name)

    def profile_iterable(self, iterable, action_name):
        iterator = iter(iterable)
        while True:
            try:
                self.start(action_name)
                value = next(iterator)
                self.stop(action_name)
                yield value
            except StopIteration:
                self.stop(action_name)
                break

    def describe(self):
        """
        Logs a profile report after the conclusion of the training run.
        """
        pass

class AdvancedProfiler(BaseProfiler):
    """
    This profiler uses Python's cProfiler to record more detailed information about
    time spent in each function call recorded during a given action. The output is quite
    verbose and you should only use this if you want very detailed reports.
    """

    def __init__(self, output_filename=None, line_count_restriction=1.0):
        """
        :param output_filename (str): optionally save profile results to file instead of printing
            to std out when training is finished.
        :param line_count_restriction (int|float): this can be used to limit the number of functions
            reported for each action. either an integer (to select a count of lines),
            or a decimal fraction between 0.0 and 1.0 inclusive (to select a percentage of lines)
        """
        self.profiled_actions = {}
        self.output_filename = output_filename
        self.line_count_restriction = line_count_restriction

    def start(self, action_name):
        if action_name not in self.profiled_actions:
            self.profiled_actions[action_name] = cProfile.Profile()
        self.profiled_actions[action_name].enable()

    def stop(self, action_name):
        pr = self.profiled_actions.get(action_name)
        if pr is None:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        pr.disable()

    def describe(self):
        self.recorded_stats = {}
        for action_name, pr in self.profiled_actions.items():
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
            ps.print_stats(self.line_count_restriction)
            self.recorded_stats[action_name] = s.getvalue()
        if self.output_filename is not None:
            # save to file
            with open(self.output_filename, "w") as f:
                for action, stats in self.recorded_stats.items():
                    f.write(f"Profile stats for: {action}")
                    f.write(stats)
        else:
            # log to standard out
            output_string = "\nProfiler Report\n"
            for action, stats in self.recorded_stats.items():
                output_string += f"\nProfile stats for: {action}\n{stats}"
            logger.info(output_string)