"""
Class to make a uniform report format.
"""

from typing import Iterable, List

import matplotlib.pyplot as plt
from crm_validator.constants import METRIC, NAME, PASSED, PLOT, REPORT


class PlotParams:
    def __init__(
        self,
        values: Iterable,
        labels: Iterable,
        kind: str,
        title: str = None
    ) -> None:
        self.values = values
        self.labels = labels
        self.kind = kind
        self.title = title


class SubReport:
    """
    This class is a basic report format.
    Can be used for plotting.
    """
    def __init__(
        self,
        passed: bool,
        metrics: dict,
        reports: dict,
        name: str = None,
        description: str = None,
        plots: List[PlotParams] = None
    ) -> None:
        self.name = name
        self.passed = passed
        self.metrics = metrics
        self.reports = reports
        self.description = description
        self.plots = []

        if plots:
            for plot in plots:
                self.plots.append(
                    self.create_plot(
                        plot
                    )
                )

    def __str__(self) -> str:
        return str(
            {
                NAME: self.name,
                PASSED: self.passed,
                METRIC: self.metrics,
                REPORT: self.reports,
                PLOT: True if self.plots else False
            }
        )

    def create_plot(
        self,
        plot_params: PlotParams
    ):
        """
        Function to create specified plots in the report.

        Parameters
        ----------
            values : list
                The list of names from the report to be plotted.
            kind : str ["pie" | "line"]
                Type of plot.
                Supported values: "pie", "line".

        Output
        ------
            A Matplotlib figure.
        """
        fig = plt.figure()

        if plot_params.kind == "pie":
            plt.pie(
                x=plot_params.values,
                labels=plot_params.labels
            )

        elif plot_params.kind == "line":
            plt.plot(
                data=plot_params.values,
                labels=plot_params.labels
            )

        elif plot_params.kind == "double_hist":
            plt.hist(
                plot_params.values[0],
                label=plot_params.labels[0]
            )
            plt.hist(
                plot_params.values[1],
                label=plot_params.labels[1]
            )
            plt.legend()

        if plot_params.title:
            plt.title(plot_params.title)

        return fig


class Report:
    def __init__(
        self,
        reports: List[SubReport],
        name: str = None
    ) -> None:
        self.name = name
        self.reports = reports

    def __str__(self) -> str:
        return str(
            {
                "name": self.name,
                "sub-reports": [
                    str(report) for report in self.reports
                ]
            }
        )

    def __iter__(self) -> List[SubReport]:
        return iter(self.reports)
