# Remote Generators — User Guide

Remote generators let you keep your usual Xopt workflow (VOCS, evaluator,
`X.step()`, `X.data`) in a notebook or Python script while the generator itself (GP training,
acquisition optimization, BAX execution paths) runs in a separate server
process. That process can be on the same machine or on a different host, such
as a GPU node.

Nothing about how you build the generator changes. You construct it exactly as in
the other Xopt examples and wrap it:

```python
generator = HTTPGenerator(base_url=SERVER_URL, generator=my_generator)
X = Xopt(generator=generator, evaluator=evaluator)
```

Your evaluator, and therefore all machine or simulation access, stays on the
client (your notebook or script). Only the generator's work moves to the
server. Everything below applies to notebooks and scripts alike; the few
practical differences are listed in
[Notebooks, scripts, and YAML](#notebooks-scripts-and-yaml).

## Which generator should I use?

There are two wrappers. They run the same optimization in the same way on the
server. The only difference is how your configured generator gets there.

- **`HTTPGenerator`** sends the generator's configuration, the same JSON/YAML
  state you get from `generator.to_json()`, and the server rebuilds it from that.
- **`CloudpickleHTTPGenerator`** sends the Python object itself with
  [cloudpickle](https://github.com/cloudpipe/cloudpickle), including the code of
  any classes you defined in your notebook or script.

The question to ask is: **can the server import every class used in my
generator?**

**Use `HTTPGenerator`** when everything in your generator comes from Xopt,
BoTorch, GPyTorch, or a package that is also installed on the server. This
covers most problems:

- A standard BO generator (`UpperConfidenceBoundGenerator`,
  `ExpectedImprovementGenerator`, `MOBOGenerator`, ...).
- `gp_constructor` options such as `use_low_noise_prior`, or a stock GPyTorch
  kernel passed through `covar_modules`, for example a `PeriodicKernel`.
- A custom prior mean or BAX algorithm that lives in an installed package. For
  example, a model used as a `mean_modules` prior, or a BAX `Algorithm` subclass:
  the server can rebuild these as long as **the same package and version are installed there**.

**Use `CloudpickleHTTPGenerator`** when part of the generator is defined in
your notebook, or in the script you are running, and isn't installed on the
server. Both count as `__main__`, so they behave the same way. The typical case is the
[custom GP model example](../single_objective_bayes_opt/custom_model.ipynb):
you write a `torch.nn.Module` prior mean in a cell, for example a thin-lens
beam-size model or a fit to last week's data, and pass it as
`mean_modules={"beam_size": MyPrior()}`. Another case is a BAX `Algorithm`
subclass you are still developing. With `HTTPGenerator` these
fail when the session is created, with an error such as
`Can't get attribute 'MyPrior' on <module '__main__'>`. Cloudpickle sends the
class definition along with the object.

Classes imported from a module on your machine (e.g. `from my_priors import
MyPrior`) are different. Both wrappers send them by import path, not as code,
so the server must be able to import the same module. If it can't, either
install the module on the server and use `HTTPGenerator`, or tell cloudpickle
to send the module's code before creating the session:

```python
import cloudpickle
import my_priors

cloudpickle.register_pickle_by_value(my_priors)
```

Once a class has stabilized, the cleanest option is to put it in a package
installed on both sides and use `HTTPGenerator`. Its plain configuration is
easier to save, reproduce, and inspect.

Some points apply to both wrappers:

- **Neither one decides whether the GPU is used.** Both run the generator on the
  server. GPU use is set on the generator as usual (`use_cuda=True`) and depends
  on the server's hardware.
- **Package versions must match.** Xopt, PyTorch, BoTorch, and GPyTorch need
  compatible versions on client and server. This matters even more for
  cloudpickle.
- **Only use servers and clients you trust.** See [Security](#security).

## 1. Install

On both the client and the server, from the Xopt repository root:

```bash
python -m pip install -e ".[remote]"
```

Also install any packages your generator needs on both sides, such as your
group's model package or a BAX algorithm library.

## 2. Start the server

For `HTTPGenerator` only:

```bash
export XOPT_HTTP_GENERATOR_TOKEN="<a long random string>"   # recommended
python -m uvicorn xopt.generators.remote.server:app \
  --host 127.0.0.1 --port 8002
```

For `CloudpickleHTTPGenerator`, start the cloudpickle server instead. It also
serves `HTTPGenerator` clients, and it will not start without a token:

```bash
export XOPT_HTTP_GENERATOR_TOKEN="<a long random string>"   # required
python -m uvicorn xopt.generators.remote.cloudpickle_server:app \
  --host 127.0.0.1 --port 8002
```

If `XOPT_HTTP_GENERATOR_TOKEN` is unset, the plain server accepts requests from
anyone who can reach the port. Sessions are held in memory and are lost when the
server restarts.

## 3. Connect from the client

```python
from xopt.generators.remote.http import HTTPGenerator
# or
from xopt.generators.remote.cloudpickle import CloudpickleHTTPGenerator

generator = HTTPGenerator(
    base_url="http://127.0.0.1:8002",
    generator=my_generator,
    api_key=token,  # the server's XOPT_HTTP_GENERATOR_TOKEN
)
```

`127.0.0.1` only works when the client and server are on the same machine. For
a server on another host, use its address, or forward the port with an SSH
tunnel (`ssh -L 8002:127.0.0.1:8002 server-host`) and keep `127.0.0.1` on the
client.

## 4. Working with the remote generator

- `X.step()`, `X.random_evaluate()` and `X.data` behave as usual. New data is
  sent to the server and candidates come back.
- **Inspecting the model.** `pull()` returns a local copy of the server's
  generator, including the fitted model. Methods that aren't part of the remote
  interface, such as `X.generator.visualize_model()`, do this automatically.
  The copy is a snapshot. Changing it does not affect the server, and you need
  to pull again after more steps. Plotting and any model evaluation on the copy
  run on the client.
- **Ending a session.** Call `X.generator.finalize()` to free the session on
  the server. The server keeps running for other sessions.

## Notebooks, scripts, and YAML

The remote generator works the same in all three. These are the practical
differences.

**Server address and token.** Read them from the environment rather than
writing them into the code:

```python
import os

generator = HTTPGenerator(
    base_url=os.environ.get("XOPT_HTTP_URL", "http://127.0.0.1:8002"),
    generator=my_generator,
    api_key=os.environ["XOPT_HTTP_GENERATOR_TOKEN"],
)
```

A script sees the variables of the shell it is started from. A Jupyter kernel
only sees variables that were set before Jupyter was launched; setting them in
a terminal afterwards has no effect until you restart Jupyter. In a notebook
you can also set them in a cell with `os.environ[...] = ...` before creating
the generator.

**Ending the session.** Each `HTTPGenerator` opens one session on the server
the first time it is used, and only `finalize()` closes it. `X.run()` calls
`finalize()` when it finishes. Nothing else does: not a `X.step()` loop, a
script that crashes or is interrupted with Ctrl-C, a kernel restart, or
re-running the notebook cell that creates the generator. Each of those leaves
the old session on the server until the server restarts. In a script with its
own loop, use `try`/`finally`:

```python
try:
    for _ in range(n_steps):
        X.step()
finally:
    X.generator.finalize()
```

In a notebook, call `X.generator.finalize()` before re-creating the generator.

**Inspecting the model.** In a notebook you call `X.generator.pull()` or
`X.generator.visualize_model()` whenever you want to look at the model. In a
script, pull once at the end, before `finalize()`, and save what you need:

```python
fig, ax = X.generator.visualize_model()
fig.savefig("model.png")
```

`X.data` is always local. `api_key` is never written to dump files.

**Dumps and the server's state.** The client holds a copy of the generator
that is only as recent as the last `pull()`. To keep dumps in step with the
server, `X.dump()`, including the automatic dump after every step when
`xopt_dump_file` is set, pulls the latest state first. This matters for
generators that carry state between steps, such as TuRBO trust regions.

- If the server can't be reached within `dump_pull_timeout` (default 10 s),
  the dump is still written from the last pulled copy, with a warning.
- Set `pull_on_dump=False` on the wrapper to skip the pull, e.g. if pulling a
  large fitted model after every step is too slow. Dumps then contain the last
  pulled copy.
- **`X.yaml()`, `X.json()` and displaying `X` in a notebook do not pull.** They
  show the last pulled copy, which can be out of date. Call
  `X.generator.pull()` first if you need the current state, or save with
  `X.dump()`.

**Reloading a dump.** A dump can be loaded again to continue the run in a
new session on the server. The token isn't saved, so add it before loading.
If the dump contains data, loading it sends that data to the server right
away, so the server must be running:

```python
import os
import yaml
from xopt import Xopt

config = yaml.safe_load(open("dump.yaml"))
config["generator"]["api_key"] = os.environ["XOPT_HTTP_GENERATOR_TOKEN"]
X = Xopt.from_dict(config)
```

To just look at the data without contacting the server, read the file
directly: `yaml.safe_load(open("dump.yaml"))["data"]`.

By default, dumps leave out PyTorch modules such as custom kernels and prior
means, so a reloaded generator quietly falls back to the default model. To
keep them, set these on the Xopt object before it writes the dump:

```python
X.serialize_torch = True
X.serialize_inline = True  # embed in the YAML; otherwise saved as .pt files in the working directory
```

Set them as attributes as shown. Passing them to `X.dump()` raises a
`TypeError`. Without `serialize_inline`, the dump refers to the `.pt` files
by a path relative to the working directory, so load it from that same
directory and keep the files with the dump.

The modules are stored by import path, the same as in
[Which generator should I use?](#which-generator-should-i-use), so a reload
only works where their classes can be imported. Classes defined in a notebook
or script cannot be reloaded from a dump. This is normal Xopt behaviour, not
specific to remote generators.

**YAML.** Use `name: http` (or `name: cloudpickle_http`) and put your usual
generator config under a nested `generator:` key. `vocs` can go at the top
level, under the outer generator, or under the nested one:

```yaml
generator:
  name: http
  base_url: http://127.0.0.1:8002
  generator:
    name: upper_confidence_bound
    beta: 0.1
  vocs:
    variables: {x1: [0, 6.28]}
    objectives: {y1: MINIMIZE}
evaluator:
  function: xopt.resources.test_functions.sinusoid_1d.evaluate_sinusoid
stopping_condition:
  name: MaxEvaluationsCondition
  max_evaluations: 20
```

Keep the token out of the file and add it when loading:

```python
import os
import yaml
from xopt import Xopt

config = yaml.safe_load(open("remote_ucb.yaml"))
config["generator"]["api_key"] = os.environ["XOPT_HTTP_GENERATOR_TOKEN"]
X = Xopt.from_dict(config)
X.run()
```

A YAML file can only reference classes by import path, so custom classes must
come from a module the server can import. Notebook-defined classes are not an
option here.

## 5. Examples

| Notebook | Wrapper | Shows |
| --- | --- | --- |
| [`http_generator_ucb.ipynb`](http_generator_ucb.ipynb) | `HTTPGenerator` | Remote UCB optimization and model visualization |
| [`http_generator_bax.ipynb`](http_generator_bax.ipynb) | `HTTPGenerator` | Remote BAX with a built-in algorithm (`GridOptimize`) |
| [`http_generator_custom_model.ipynb`](http_generator_custom_model.ipynb) | `CloudpickleHTTPGenerator` | Custom kernel and a notebook-defined prior mean |

Each notebook reads the server address from an environment variable (see
[Notebooks, scripts, and YAML](#notebooks-scripts-and-yaml)). Check the first
cell for the variable name. The same code works as a script.

## Security

- **Both servers will run code sent by any client that can authenticate.**
  Cloudpickle does this by design. `HTTPGenerator` also does it: PyTorch modules
  in the configuration (custom kernels and mean functions) are sent as
  `torch.save` data and loaded on the server with full pickle. Treat any client
  that can reach the server as able to run code there.
- Always set `XOPT_HTTP_GENERATOR_TOKEN`, bind to `127.0.0.1` and reach the
  server through an SSH tunnel or a restricted network, and use TLS when
  crossing networks. Do not expose either server publicly.
- The client also unpickles whatever the server returns from `pull()`, so only
  connect to servers you trust.
- For reproducible results, record the Xopt version and the versions of your
  own packages on both client and server.

## Running the tests

From the Xopt repository root:

```bash
python -m pytest xopt/tests/generators/remote -v --no-cov
```

The tests start their own server, so you don't need one running on port 8002.
