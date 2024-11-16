import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_table(fname: str, a_col: str, b_col: str):
    df = pd.read_csv(fname)
    a = df[[a_col]].to_numpy().flatten()
    b = df[[b_col]].to_numpy().flatten()

    return a, b


def plot(
    xs_main,
    ys,
    plots: list,
    title: str,
    xlabel: str,
    ylabel: str,
):
    plt.plot(
        xs_main, ys, linestyle="None", marker="o", color="b", label="data", markersize=2
    )
    for xs, coeffs, r2, yhat, color, formatter in plots:
        plt.plot(
            xs,
            yhat,
            linestyle="--",
            color=color,
            label=formatter(coeffs, r2),
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def degn_fit(xs, ys, n):
    coeffs = np.polyfit(xs, ys, n)

    p = np.poly1d(coeffs)

    yhat = p(xs)
    ybar = np.sum(ys) / len(ys)
    ss_res = np.sum((ys - yhat) ** 2)
    ss_tot = np.sum((ys - ybar) ** 2)
    r2 = 1 - ss_res / ss_tot

    rmse = np.sqrt(ss_res / len(xs))

    return coeffs, rmse, ss_res, r2, p


def gold_price_vs_gdp_share():
    _, price = read_table(
        "datasets\dataset_annual.csv",
        a_col="Date",
        b_col="Price",
    )
    price = [float(str(p).replace(",", "")) for p in price]

    _, gdp_share = read_table(
        "datasets\dataset_annual.csv",
        a_col="Date",
        b_col="US GDP Share %",
    )

    coeffs_l, rmse_l, ss_res_l, r2_l, fit_l = degn_fit(gdp_share, price, 1)
    coeffs_p, rmse_p, ss_res_p, r2_p, fit_p = degn_fit(gdp_share, price, 2)
    coeffs_p3, rmse_p3, ss_res_p3, r2_p3, fit_p3 = degn_fit(gdp_share, price, 3)

    print("linear:", "RMSE:", round(rmse_l, 4), "SS_RES:", round(ss_res_l, 4))
    print("poly_2:", "RMSE:", round(rmse_p, 4), "SS_RES:", round(ss_res_p, 4))
    print("poly_3:", "RMSE:", round(rmse_p3, 4), "SS_RES:", round(ss_res_p3, 4))

    synth = np.arange(start=min(gdp_share), stop=max(gdp_share) + 0.5, step=0.25)

    plot(
        gdp_share,
        price,
        [
            (
                synth,
                coeffs_l,
                r2_l,
                fit_l(synth),
                "r",
                lambda a, b: f"linear_regression: {round(a[0],4)}x + {round(a[1],4)}, R^2: {round(b,4)}",
            ),
            (
                synth,
                coeffs_p,
                r2_p,
                fit_p(synth),
                "g",
                lambda a, b: f"poly_2_regression: {round(a[0],4)}x^2 + {round(a[1],4)}x + {round(a[2],4)}, R^2: {round(b,4)}",
            ),
            (
                synth,
                coeffs_p3,
                r2_p3,
                fit_p3(synth),
                "black",
                lambda a, b: f"poly_3_regression: {round(a[0],4)}x^3 + {round(a[1],4)}x^2 + {round(a[2],4)}x + {round(a[3],4)}, R^2: {round(b,4)}",
            ),
        ],
        title="Gold Price vs. US GDP Share %",
        xlabel="US GDP Share %",
        ylabel="Price ($/oz)",
    )


def gold_price_vs_cpi():
    _, price = read_table(
        "datasets\dataset_annual.csv",
        a_col="Date",
        b_col="Price",
    )
    price = [float(str(p).replace(",", "")) for p in price]

    _, cpi = read_table(
        "datasets\dataset_annual.csv",
        a_col="Date",
        b_col="CPIAUCSL",
    )

    coeffs_l, rmse_l, ss_res_l, r2_l, fit_l = degn_fit(cpi, price, 1)
    coeffs_p, rmse_p, ss_res_p, r2_p, fit_p = degn_fit(cpi, price, 2)
    coeffs_p4, rmse_p4, ss_res_p4, r2_p4, fit_p4 = degn_fit(cpi, price, 4)

    print("linear:", "RMSE:", round(rmse_l, 4), "SS_RES:", round(ss_res_l, 4))
    print("poly_2:", "RMSE:", round(rmse_p, 4), "SS_RES:", round(ss_res_p, 4))
    print("poly_4:", "RMSE:", round(rmse_p4, 4), "SS_RES:", round(ss_res_p4, 4))

    synth = np.arange(start=min(cpi), stop=max(cpi) + 0.5, step=0.25)

    plot(
        cpi,
        price,
        [
            (
                synth,
                coeffs_l,
                r2_l,
                fit_l(synth),
                "r",
                lambda a, b: f"linear_regression: {round(a[0],4)}x + {round(a[1],4)}, R^2: {round(b,4)}",
            ),
            (
                synth,
                coeffs_p,
                r2_p,
                fit_p(synth),
                "g",
                lambda a, b: f"poly_2_regression: {round(a[0],4)}x^2 + {round(a[1],4)}x + {round(a[2],4)}, R^2: {round(b,4)}",
            ),
            (
                synth,
                coeffs_p4,
                r2_p4,
                fit_p4(synth),
                "black",
                lambda a, b: f"poly_4_regression: {round(a[0],8)}x^4 + {round(a[1],8)}x^3 + {round(a[2],8)}x^2 + {round(a[3],4)}x + {round(a[4],4)}, R^2: {round(b,4)}",
            ),
        ],
        title="Gold Price vs. CPI",
        xlabel="CPI",
        ylabel="Price ($/oz)",
    )


if __name__ == "__main__":
    gold_price_vs_gdp_share()
    gold_price_vs_cpi()
