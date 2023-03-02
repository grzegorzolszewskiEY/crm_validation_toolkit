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
    It can be used for plotting (see `PlotParams`).

    Attributes
    ----------
        `metrics` : dict
            Holds pairs of metric_name : metric_value.
            These are specified metrics to be reported.
        `reports` : dict
            Holds pairs of report_name : report_value.
            These are values to be reported other than those in
            `metrics`.
        `name` : str [OPTIONAL]
            Name of the sub-report.
        `description` : str [OPTIONAL]
            Description of the sub-report.
        `plots` : list[PlotParams] [OPTIONAL]
            A list of PlotParams for each plot to be made as
            part of the report.
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
        self.description = description
        self.passed = passed
        self.metrics = metrics
        # TODO : Support matrix reports
        self.reports = reports
        self.plots = []

        if plots:
            for plot in plots:
                self.plots.append(
                    self.create_plot(
                        plot
                    )
                )

    def __str__(self) -> str:
        """
        This function allows Python to represent an object of
        this class as a `str`.
        """
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
            plot_params : PlotParams
                An object containing values, labels, and kind of plot
                to be generated.

        Output
        ------
            A Matplotlib figure.
        """
        plt.ioff()
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

        plt.close()
        return fig


class Report:
    """
    This class is used to create an overall report for
    a test.

    Attributes
    ----------
        subreports : List[SubReport]
            A list of sub-reports that is part of this
            report. This is useful in cases where reports for
            a validation test are generated per facility grade,
            for instance. Then, each of those can be a
            sub-report for the validation test.
        name : str [OPTIONAL]
            The name of the report.
        description : str [OPTIONAL]
            A description of the report.
    """
    def __init__(
        self,
        subreports: List[SubReport],
        name: str = None,
        description: str = None
    ) -> None:
        self.name = name
        self.description = description
        self.subreports = subreports

    def __str__(self) -> str:
        return str(
            {
                "name": self.name,
                "description": self.description,
                "sub-reports": [
                    str(report) for report in self.subreports
                ]
            }
        )

    def print_output(self) -> str:
        output_string = ""
        # Display report heading
        output_string += f"{'='*len(self.name)}" + "\n"
        output_string += f"{self.name.upper()}" + "\n"
        output_string += f"{'='*len(self.name)}" + "\n"

        # Display description
        if self.description is not None:
            output_string += f"{self.description}" + "\n"
            output_string += "\n"

        # Display sub-reports
        for subreport in self.subreports:
            output_string += "\n"
            indent_level = 1

            # Display sub-report title
            if subreport.name is not None:
                output_string += "\t"*indent_level
                output_string += f"{subreport.name.upper()}" + "\n"
                output_string += "\t"*indent_level
                output_string += f"{'-'*len(subreport.name)}" + "\n"
                indent_level = 2

            # Display description
            if subreport.description is not None:
                output_string += "\t"*indent_level
                output_string += f"{subreport.description}" + "\n"
                output_string += "\n"
                indent_level = 2

            output_string += "\t"*indent_level
            output_string += "Metrics\n" + "\t" * indent_level + "-------\n"
            # Display metrics
            for metric in subreport.metrics:
                output_string += "\t"*indent_level
                output_string += f"{metric}\t:{subreport.metrics[metric]}\n"

            output_string += "\n"
            output_string += "\t"*indent_level
            output_string += "Report Values\n"
            output_string += "\t" * indent_level + "-------------" + "\n"
            # Display reported values
            for value in subreport.reports:
                output_string += "\t"*indent_level
                output_string += f"{value}\t:{subreport.reports[value]}" + "\n"

        output_string += f"{'='*len(self.name)}" + "\n\n"
        return output_string

    def __iter__(self) -> List[SubReport]:
        return iter(self.subreports)
