import plotnine as p9

kst_theme = p9.theme_bw() + p9.theme(plot_title=p9.element_text(ha='center'))


class BasePlots:
    def __init__(
            self, data_set, main_title="", x_label="", y_label="", legend_title="", theme=kst_theme
    ):
        self.main_title = main_title
        self.x_label = x_label
        self.y_label = y_label
        self.legend_title = legend_title
        self._plot = (
                p9.ggplot(data_set)
                + theme
        )

    @property
    def plot(self):
        return self._plot + p9.labs(x=self.x_label, y=self.y_label, title=self.main_title, color=self.legend_title)


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
            theme=kst_theme,
            line=None,
    ):
        super().__init__(data_set, main_title, x_label, y_label, legend_title, theme)
        self._plot = self._plot + p9.aes(**aes) + p9.geom_point(**points_features)
        if line is not None:
            self._plot = self._plot + line

    def add_gg_object(self, gg_object):
        self._plot = self._plot + gg_object
