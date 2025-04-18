import numpy as np


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
    rt_scenarios = (
        ("decreasing", 42, 1.0, 0.9),
        ("increasing", 43, 1.0, 1.1),
        ("notrend", 44, 1.0, 1.0),
    )

    for i in range(len(rt_scenarios)):
        weekly_rt = generate_rt_scenario(
            rt_scenarios[i][2],
            rt_scenarios[i][3],
            36,
            16,
            0.01,
            0.5,
            np.random.default_rng(1),
        )

        with open(
            f"pipeline/input/rt/{rt_scenarios[i][0]}.txt", "w"
        ) as outfile:
            outfile.write("\n".join([str(rt) for rt in weekly_rt]))
