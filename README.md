# cfa-genomic-lag-effect

⚠️ This is a work in progress ⚠️

This repo contains code to study via simulations the effect of lag (from sample collection to sequence availability) on the utility of genomic data for estimating epidemiological parameters.
It consists of a small python package, `lag`, and some associated utilities in `/scripts`.

## Getting started

This is a [poetry](https://python-poetry.org/) project with a split codebase.
The more general (and generalizable) modeling code is in the python package `lag`, which can be installed with `poetry install`.
The simulation study also uses more special-purpose code (e.g., for approximating observed distributions of reporting lags), which live as callable python scripts in `pipeline/`.

The pipeline is implemented in [snakemake](https://snakemake.github.io/) (which will also be installed with `poetry install`).
To run the pipeline, you will need to:
1. Download and uncompress Nextstrain's open [metadata.tsv](https://docs.nextstrain.org/projects/ncov/en/latest/reference/remote_inputs.html) for SARS-CoV-2. It can be placed anywhere.
2. Create `scripts/config.json` by copying `scripts/example_config.json` and amending the `nextstrain_path` argument to point to (1).
3. `poetry run snakemake` (using the ` -j1` flag is recommended when using multiple chains in NumPyro).

Running the pipeline will take some time, as 100 replicate datasets are simulated and analyzed for each combination of:
- 3 $R_t$ trends: a scenario with no trend where $R_t \approx 1$ for the entire time period of interest, a scenario where $R_t$ decreases in the most recent months, and a scenario where $R_t$ increases in the most recent months.
- 3 initial counts of incident infections: 1,000, 2,000, and 4,000.
- 5 distributions on the lag between sample collection and sequence availability. First, we obtain $\hat{f}(\ell)$, a [spline-based approximation](https://www.sciencedirect.com/science/article/pii/S0377042724003807) to the empirical probability density function of lags $\ell$ in the Nextstrain open data from 2020 to 2025. Then when simulating data we use $g(\ell) = k \ell$ for $k \in \{0, 1/4, 1/2, 3/4, 1\}$ to interpolate between a regime of instantaneous data availability and current reality.

### Visualizing the simulated scenarios

The command `poetry run snakemake diagnostics` will produce plots showing:
- The 3 $R_t$ scenarios (in `pipeline/out/rt/`).
- The 9 pairs of incidence and prevalence curves resulting from each $R_t$ scenario and initial incident infection count (in `pipeline/out/infections/`).
- The scaled lag distributions (in `pipeline/out/lag/`) each a multi-panel plot showing
  - The probability density function (minus the long upper 5\% tail).
  - The cumulative distribution function (for the entire distribution).
  - A comparison of the approximating distribution to the samples on which it was fit.
  - A comparison of samples of the approximation to the approximation itself.

Note that if snakemake has not yet been called, this will run some of the preliminary pipeline steps.

## Project Admin

Andy Magee, PhD, (@afmagee42)

------------------------------------------------------------------------------------

## Disclaimers

### General Disclaimer

This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm). GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

### Public Domain Standard Notice

This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

### License Standard Notice

This repository is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

### Privacy Standard Notice

This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

### Contributing Standard Notice

Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

### Records Management Standard Notice

This repository is not a source of government records but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).
