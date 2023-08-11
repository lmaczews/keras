import plotnine as p9


class BasePlots:
    def __init__(
        self, data_set, main_title, x_label, y_label, legend_title="", theme=p9.theme_bw
    ):
        self._plot = (
            p9.ggplot(data_set)
            + p9.ggtitle(main_title)
            + p9.xlab(x_label)
            + p9.ylab(y_label)
            + theme()
        )

    @property
    def plot(self):
        return self._plot


class ScatterPlot(BasePlots):
    def __init__(
        self,
        data_set,
        aes,
        points_features={},
        main_title="",
        x_label="",
        y_label="",
        legend_title="",
        theme=p9.theme_bw,
        line=None,
    ):
        super().__init__(data_set, main_title, x_label, y_label, legend_title, theme)
        self._plot = self._plot + p9.aes(**aes) + p9.geom_point(**points_features)
        if line is not None:
            self._plot = self._plot + line

    def add_gg_object(self, gg_object):
        self._plot = self._plot + gg_object