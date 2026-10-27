# Remote Generators — User Guide

Remote generators let you keep your usual Xopt workflow in a notebook (VOCS,
evaluator, `X.step()`, `X.data`) while the generator itself (GP training,
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
client. Only the generator's work moves to the server.

## Which generator should I use?

There are two wrappers. They run the same optimization in the same way on the
server. The only difference is how your configured generator gets there.

- **`HTTPGenerator`** sends the generator's configuration, the same JSON/YAML
  state you get from `generator.to_json()`, and the server rebuilds it from that.
- **`CloudpickleHTTPGenerator`** sends the Python object itself with
  [cloudpickle](https://github.com/cloudpipe/cloudpickle), including the code of
  any classes you defined in the notebook.

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
your notebook and isn't installed on the server. The typical case is the
[custom GP model example](../single_objective_bayes_opt/custom_model.ipynb):
you write a `torch.nn.Module` prior mean in a cell, for example a thin-lens
beam-size model or a fit to last week's data, and pass it as
`mean_modules={"beam_size": MyPrior()}`. Another case is a BAX `Algorithm`
subclass you are still developing in the notebook. With `HTTPGenerator` these
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

## 3. Connect from the notebook

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

## 5. Examples

| Notebook | Wrapper | Shows |
| --- | --- | --- |
| [`http_generator_ucb.ipynb`](http_generator_ucb.ipynb) | `HTTPGenerator` | Remote UCB optimization and model visualization |
| [`http_generator_bax.ipynb`](http_generator_bax.ipynb) | `HTTPGenerator` | Remote BAX with a built-in algorithm (`GridOptimize`) |
| [`http_generator_custom_model.ipynb`](http_generator_custom_model.ipynb) | `CloudpickleHTTPGenerator` | Custom kernel and a notebook-defined prior mean |

Each notebook reads the server address from an environment variable, which you
can set before launching Jupyter. Check the first cell for the variable name.

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
