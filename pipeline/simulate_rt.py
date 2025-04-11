import numpy as np

from pipeline.utils import construct_seed, parser, read_config


def ar1(
    mu: np.typing.NDArray,
    sd: float,
    ac: float,
    rng: np.random.Generator,
):
    """
    Draw from an AR1 process with mean vector mu,
    standard deviation sd, and autocorrelation ac.
    """
    n = mu.shape[0]
    z = rng.normal(loc=0.0, scale=1.0, size=n) * sd
    x = np.zeros(n)
    x[0] = z[0]
    for i in range(1, n):
        x[i] = x[i - 1] * ac + z[i]
    return np.exp(np.log(mu) + x)


def generate_rt_scenario(
    r_init: float,
    r_final: float,
    init_weeks: int,
    change_weeks: int,
    sd: float,
    ac: float,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Generates an Rt time series from an AR1 process where for `init_weeks` Rt
    has median `r_init`, then the median changes towards `r_final` over the
    course of `change_weeks`.
    """
    mean = np.concat(
        (
            np.array([r_init] * init_weeks),
            np.linspace(r_init, r_final, change_weeks),
        )
    )

    return ar1(mean, sd, ac, rng)


if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)

    rt_scenarios = {
        "decreasing": (
            config["simulations"]["r_med"],
            config["simulations"]["r_low"],
        ),
        "constant": (
            config["simulations"]["r_med"],
            config["simulations"]["r_med"],
        ),
        "increasing": (
            config["simulations"]["r_med"],
            config["simulations"]["r_high"],
        ),
    }

    seed = construct_seed(
        config["seed"],
        scenario=args.scenario,
        i0=None,
        scaling_factor=None,
        rep=None,
    )
    weekly_rt = generate_rt_scenario(
        rt_scenarios[args.scenario][0],
        rt_scenarios[args.scenario][1],
        config["simulations"]["n_init_weeks"],
        config["simulations"]["n_change_weeks"],
        config["simulations"]["r_sd"],
        config["simulations"]["r_ac"],
        np.random.default_rng(seed),
    )

    with open(args.outfile, "w") as outfile:
        outfile.write("\n".join([str(rt) for rt in weekly_rt]))
