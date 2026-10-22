# Third-party notices

This package contains code derived from the following MIT-licensed works. Their notices are reproduced here.

## Noise (qlabs-eng/noise) and the Relay script (erodola/noise, PR #20)

`relay_opal/relay/kernels.py`, `relay_opal/relay/estimator.py` and `relay_opal/models/noise_gpt.py` are derived from
`train/val/run.py` (Relay, Emanuele Rodolà; CutBI, Euraba; Boost -> Cut, spicylemonade) and `utils/model.py`,
`utils/eval.py`, `utils/prepare_data.py` of https://github.com/qlabs-eng/noise. `relay_opal/fidelity.py` is derived
from `check_relay.py` of https://github.com/erodola/noise/releases/tag/relay-track1-v1.

```
MIT License

Copyright (c) 2026 qlabs-eng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OPAL (xingjian-bai/opal)

OPAL is a dependency (`opal-optimizer` on PyPI), not vendored. The starting program files in
`relay_opal/opal/program/` and the prompt template in `relay_opal/opal/prompt.md` follow the structure of OPAL's
`opal/programs.py` and `opal/prompts/neutral.md` (MIT, Copyright (c) 2026 Xingjian Bai, Jia-Hao Liang, Jayden Teoh, Vincent Sitzmann).
