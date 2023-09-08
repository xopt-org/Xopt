import numpy as np
import pytest
from pydantic import ValidationError

from xopt import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.es.extremumseeking import ExtremumSeekingGenerator
from xopt.resources.testing import TEST_VOCS_BASE
from xopt.vocs import VOCS


class TestExtremumSeekingGenerator:
    def test_es_generate_multiple_points(self):
        gen = ExtremumSeekingGenerator(vocs=TEST_VOCS_BASE)

        # Try to generate multiple samples
        with pytest.raises(NotImplementedError):
            gen.generate(2)

    def test_es_options(self):
        gen = ExtremumSeekingGenerator(vocs=TEST_VOCS_BASE)

        with pytest.raises(ValidationError):
            gen.k = "yo"

        with pytest.raises(ValidationError):
            gen.oscillation_size = 0.0

        with pytest.raises(ValidationError):
            gen.decay_rate = -1.0

        gen.oscillation_size = 0.2
        gen.decay_rate = 2
        assert gen.oscillation_size == 0.2
        assert gen.decay_rate == 2

    def test_es_agreement(self):
        """Compare the first 100 steps between Vanilla ES and Xopt ES"""
        # Vanilla ES

        # This is a very simple demonstration of how to apply the ES method
        # The following example is a 4 parameter time-varying system

        np.random.seed(42)  # set deterministic run

        # Total number of ES steps to take
        ES_steps = 100

        # For the parameters being tuned, the first step is to
        # define upper and lower bounds for each parameter

        # Upper bounds on tuned parameters
        p_max = 2 * np.ones(10)

        # Lower bounds on tuned parameters
        p_min = -2 * np.ones(10)

        # Number of parameters being tuned
        nES = len(p_min)

        # Average values for normalization
        p_ave = (p_max + p_min) / 2.0

        # Difference for normalization
        p_diff = p_max - p_min

        # Function that normalizes paramters
        def p_normalize(p):
            p_norm = 2.0 * (p - p_ave) / p_diff
            return p_norm

        # Function that un-normalizes parameters
        def p_un_normalize(p):
            p_un_norm = p * p_diff / 2.0 + p_ave
            return p_un_norm

        # Normalization allows you to easily handle a group of parameters
        # which might have many orders of magnitude difference between their values
        # with this normalization the normalized values live in [-1,1]

        # Now we define some ES parameters

        # This keeps track of the history of all of the parameters being tuned
        pES = np.zeros([ES_steps, nES])

        # Start with initial conditions inside of the max/min bounds
        # In this case I will start them near the center of the range
        pES[0] = p_ave

        # This keeps track of the history of all of the normalized parameters being tuned
        pES_n = np.zeros([ES_steps, nES])

        # Calculate the mean value of the initial condtions
        pES_n[0] = p_normalize(pES[0])

        # This keeps track of the history of the measured cost function
        cES = np.zeros(ES_steps)

        # This is the unknown time-varying function being minimized
        # For applications to a real system once parameters are set some kind of measure
        # of performance is returned to the ES algorithm in place of this example function

        # This is the unknown optimal point
        p_opt = 1.5 * (2 * np.random.rand(nES) - 1)
        # Various frequencies for unknown points
        w_opt = 0.25 + 2 * np.random.rand(nES)

        def f_ES_minimize(p):
            # This simple cost will be distance from the optimal point
            f_val = np.sum(p**2)
            return f_val

        # Calculate the initial cost function value based on initial conditions
        cES[0] = f_ES_minimize(pES[0])

        # These are the unknown optimal values (just for plotting)
        p_opt_t = np.zeros([ES_steps, nES])
        for n in np.arange(nES):
            p_opt_t[:, n] = p_opt[n] * (
                1 + np.sin(2 * np.pi * w_opt[n] * np.arange(ES_steps) / 2000)
            )

        # ES dithering frequencies, for iterative applications the dithering frequencies
        # are simply uniformly spread out between 1.0 and 1.75 so that no two
        # frequencies are integer multiples of each other
        wES = np.linspace(1.0, 1.75, int(np.ceil(nES / 2)))

        # ES dt step size, this particular choice of dtES ensures that the parameter
        # oscillations are smooth with at least 10 steps required to complete
        # one single sine() or cosine() oscillation when the gain kES = 0
        dtES = 2 * np.pi / (10 * np.max(wES))

        # ES dithering size
        # In normalized space, at steady state each parameter will oscillate
        # with an ampltidue of \sqrt{aES/wES}, so for example, if you'd like
        # the parameters to have normalized osciallation sizes you
        # choose the aES as:
        oscillation_size = 0.1
        aES = np.zeros(nES)
        for n in np.arange(nES):
            jw = int(np.floor(n / 2))
            aES[n] = wES[jw] * (oscillation_size) ** 2

        # Note that each parameter has its own frequency and its own oscillation size

        # ES feedback gain kES (set kES<0 for maximization instead of minimization)
        kES = 2.0

        # The values of aES and kES will be different for each system, depending on the
        # detailed shape of the functions involved, an intuitive way to set these ES
        # parameters is as follows:
        # Step 1: Set kES = 0 so that the parameters only oscillate about their initial points
        # Step 2: Slowly increase aES until parameter oscillations are big enough to cause
        # measurable changes in the noisy function that is to be minimized or maximized
        # Step 3: Once the oscillation amplitudes, aES, are sufficiently big, slowly increase
        # the feedback gain kES until the system starts to respond. Making kES too big
        # can destabilize the system

        # Decay rate. This value is optional, it causes the oscillation sizes to naturally decay.
        # If you want the parameters to persistently oscillate without decay, set decay_rate = 1.0
        decay_rate = 1.0

        # Decay amplitude (this gets updated by the decay_rate to lower oscillation sizes
        amplitude = 1.0

        # This function defines one step of the ES algorithm at iteration i
        def ES_step(p_n, i, cES_now, amplitude):
            # ES step for each parameter
            p_next = np.zeros(nES)

            # Loop through each parameter
            for j in np.arange(nES):
                # Use the same frequency for each two parameters
                # Alternating Sine and Cosine
                jw = int(np.floor(j / 2))
                if (-1) ** j > 0:
                    p_next[j] = (
                        p_n[j]
                        + amplitude
                        * dtES
                        * np.cos(dtES * i * wES[jw] + kES * cES_now)
                        * (aES[j] * wES[jw]) ** 0.5
                    )
                else:
                    p_next[j] = (
                        p_n[j]
                        + amplitude
                        * dtES
                        * np.sin(dtES * i * wES[jw] + kES * cES_now)
                        * (aES[j] * wES[jw]) ** 0.5
                    )

                # For each new ES value, check that we stay within min/max constraints
                if p_next[j] < -1.0:
                    p_next[j] = -1.0
                if p_next[j] > 1.0:
                    p_next[j] = 1.0

            # Return the next value
            return p_next

        # Now we start the ES loop
        for i in np.arange(ES_steps - 1):
            # Normalize previous parameter values
            pES_n[i] = p_normalize(pES[i])

            # Take one ES step based on previous cost value
            pES_n[i + 1] = ES_step(pES_n[i], i, cES[i], amplitude)

            # Un-normalize to physical parameter values
            pES[i + 1] = p_un_normalize(pES_n[i + 1])

            # Calculate new cost function values based on new settings
            cES[i + 1] = f_ES_minimize(pES[i + 1])

            # Decay the amplitude
            amplitude = amplitude * decay_rate

        # Xopt ES
        nES = 10

        variables = {}
        for i in range(nES):
            variables[f"p{i}"] = [-2, 2]

        vocs = VOCS(
            variables=variables,
            objectives={"f": "MINIMIZE"},
        )

        np.random.seed(42)  # set deterministic run

        ES_steps = 100

        states = {"count": 0}

        def f_ES_minimize(input_dict):
            p = []
            for i in range(10):
                p.append(input_dict[f"p{i}"])
            p = np.array(p)

            # This simple cost will be distance from the optimal point
            f_val = np.sum(p**2)

            states["count"] += 1
            outcome_dict = {"f": f_val}

            return outcome_dict

        evaluator = Evaluator(function=f_ES_minimize)
        generator = ExtremumSeekingGenerator(vocs=vocs)
        X = Xopt(vocs=vocs, evaluator=evaluator, generator=generator)

        for i in range(ES_steps):
            X.step()

        assert np.all(
            cES == X.data["f"].to_numpy()
        ), "Xopt ES does not match the vanilla one"
