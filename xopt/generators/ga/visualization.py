import copy
import glob
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from matplotlib.backend_bases import MouseButton
from scipy.optimize import curve_fit


class SnappingCursor:
    """
    A cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.
    """

    def __init__(self, fig, ax, line_list):
        self.line_colors = [l.get_color() for l in line_list]
        self.ax = ax
        self.fig = fig
        self.canvas = self.fig.canvas
        self.x = [l.get_data()[0] for l in line_list]
        self.y = [l.get_data()[1] for l in line_list]
        self._last_index = None
        self.text = ax.text(
            0.5,
            0.5,
            "",
            color="white",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
            horizontalalignment="left",
        )
        self.circ = plt.Rectangle(
            (np.nan, np.nan), 0.1, 0.1, facecolor="white", edgecolor="r", alpha=0.5
        )
        self.ax.add_patch(self.circ)
        self.screen_size = ax.get_figure().get_size_inches()
        self.pos = [0, 0]
        self.data_index = 0
        self.which_line = 0

    def set_visible(self, visible):
        self.text.set_visible(visible)
        self.circ.set_visible(visible)

    def on_mouse_move(self, event):
        if (event.inaxes != self.ax) or (event.button == MouseButton.RIGHT):
            self._last_index = None
            self.set_visible(False)
        else:
            self.set_visible(True)
            x, y = event.xdata, event.ydata

            xl = self.ax.get_xlim()
            yl = self.ax.get_ylim()

            x_rescale = self.screen_size[0] / (xl[1] - xl[0])
            y_rescale = self.screen_size[1] / (yl[1] - yl[0])

            closest_dudes = [
                ((self.x[ind] - x) * x_rescale) * ((self.x[ind] - x) * x_rescale)
                + ((self.y[ind] - y) * y_rescale) * ((self.y[ind] - y) * y_rescale)
                for ind in np.arange(0, len(self.x))
            ]
            the_best_around_index = [
                np.argmin(closest_dudes[ind]) for ind in np.arange(0, len(self.x))
            ]
            the_best_around = [
                closest_dudes[ind][the_best_around_index[ind]]
                for ind in np.arange(0, len(self.x))
            ]
            which_line = np.argmin(the_best_around)
            index = the_best_around_index[which_line]

            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            x = self.x[which_line][index]
            y = self.y[which_line][index]

            ss = 0.025
            w = (xl[1] - xl[0]) * ss * self.screen_size[1] / self.screen_size[0]
            h = (yl[1] - yl[0]) * ss
            self.circ.set_width(w)
            self.circ.set_height(h)
            self.circ.xy = (x - 0.5 * w, y - 0.5 * h)
            self.circ.set_edgecolor(self.line_colors[which_line])

            xt = (x + 1.5 * w - xl[0]) / (xl[1] - xl[0])
            yt = (y - 0.5 * h - yl[0]) / (yl[1] - yl[0])

            if xt < 0.5:
                self.text.set_horizontalalignment("left")
            else:
                self.text.set_horizontalalignment("right")
                xt = xt - 3 * w / (xl[1] - xl[0])

            self.text.set_position((xt, yt))
            self.text.set_text(f"({x:.3g}, {y:.3g})")
            self.text.set_bbox(
                dict(
                    edgecolor=self.line_colors[which_line],
                    facecolor=self.line_colors[which_line],
                    alpha=0.7,
                )
            )

            self.pos = [x, y]
            self.data_index = index
            self.which_line = which_line


class CNSGA_GUI:
    def __init__(self, vocs, pop_directory):
        self.default_color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.color_dict = {}
        self.default_legend_dict = {}
        self.legend_dict = {}
        self.gpt_data = None
        self.vocs = vocs

        self.pop_directory = pop_directory
        self.colorbar_instance = None

        self.mouse_event_handler_1 = None
        self.mouse_event_handler_2 = None

        obj = self.vocs.objective_names
        vars = self.vocs.variable_names
        self.params_from_xopt = obj + vars

        if self.vocs.constraints:
            cons = self.vocs.constraint_names
            self.params_from_xopt = self.params_from_xopt + cons

        self.wildcard_str = widgets.Text(
            value="*pop*.csv",
            placeholder="File wildcard",
            description="",
            disabled=False,
            layout=widgets.Layout(width="200px", height="30px"),
        )

        file_list = self.make_file_list()

        # Outermost container object
        self.gui = widgets.VBox(layout={"border": "1px solid grey"})

        # Layouts
        layout_150px = widgets.Layout(width="150px", height="30px")
        layout_20px = widgets.Layout(width="20px", height="30px")
        label_layout = layout_150px

        # Make plotting region
        dpi = 120
        plot_width = 500
        plot_height = 500

        plt.ioff()  # turn off interactive mode so figure doesn't show
        self.fig, self.ax = plt.subplots(
            layout="constrained", dpi=dpi, figsize=[plot_width / dpi, plot_height / dpi]
        )  # layout = constrained, tight
        plt.ion()

        self.fig.canvas.toolbar_position = "right"
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.resizable = False

        fig_hbox = widgets.HBox([self.fig.canvas], layout=widgets.Layout(width="650px"))

        # Make input region
        input_hbox = widgets.HBox()
        file_vbox = widgets.VBox()

        self.file_select = widgets.SelectMultiple(
            options=file_list,
            value=[file_list[0]],
            disabled=False,
            layout=widgets.Layout(width="500px", height="300px"),
        )
        self.active_file = widgets.Dropdown(
            options=[file_list[0]],
            disabled=False,
            layout=widgets.Layout(width="250px", height="30px"),
        )
        self.active_color = widgets.ColorPicker(
            concise=True,
            description="",
            value=self.default_color_list[0],
            disabled=False,
            layout=widgets.Layout(width="50px", height="30px"),
        )

        self.legend_checkbox = widgets.Checkbox(
            value=False,
            description="Legend: ",
            disabled=False,
            indent=False,
            layout=widgets.Layout(width="70px", height="30px"),
        )
        self.legend_str = widgets.Text(
            value="",
            placeholder="Legend name",
            description="",
            disabled=False,
            layout=widgets.Layout(width="110px", height="30px"),
        )

        self.show_constraint_violators_checkbox = widgets.Checkbox(
            value=False,
            description="Show constraint violators",
            disabled=False,
            indent=False,
            layout=widgets.Layout(width="250px", height="30px"),
        )
        data_filtering_hbox = widgets.HBox()
        data_filtering_hbox.children += (self.show_constraint_violators_checkbox,)

        self.color_fading_checkbox = widgets.Checkbox(
            value=False,
            description="Fading colors",
            disabled=False,
            indent=False,
            layout=widgets.Layout(width="125px", height="30px"),
        )
        self.color_fading_alpha = widgets.Text(
            value="0.05",
            placeholder="",
            description="alpha: ",
            disabled=False,
            layout=widgets.Layout(width="150px", height="30px"),
        )
        color_fading_hbox = widgets.HBox()
        color_fading_hbox.children += (self.color_fading_checkbox,)
        color_fading_hbox.children += (self.color_fading_alpha,)

        self.run_gpt_button = widgets.Button(
            description="Run Settings",
            disabled=True,
            button_style="",
            tooltip="Click me",
            icon="",
        )
        self.run_gpt_button.on_click(self.run_gpt)
        self.settings_menu = widgets.Dropdown(
            disabled=True, layout=widgets.Layout(width="150px", height="30px")
        )
        self.settings_value = widgets.Text(
            value="",
            placeholder="",
            disabled=True,
            layout=widgets.Layout(width="110px", height="30px"),
        )
        self.save_run_checkbox = widgets.Checkbox(
            value=False,
            description="Save run",
            disabled=False,
            indent=False,
            layout=widgets.Layout(width="80px", height="30px"),
        )

        gpt_run_hbox = widgets.HBox()
        gpt_run_hbox.children += (self.run_gpt_button,)
        gpt_run_hbox.children += (self.settings_menu,)
        gpt_run_hbox.children += (self.settings_value,)
        gpt_run_hbox.children += (self.save_run_checkbox,)

        self.best_n_checkbox = widgets.Checkbox(
            value=False,
            description="Select best N individuals: ",
            disabled=False,
            indent=False,
            layout=widgets.Layout(width="180px", height="30px"),
        )
        self.best_n_value = widgets.Text(
            value="",
            placeholder="",
            disabled=False,
            layout=widgets.Layout(width="50px", height="30px"),
        )
        self.best_n_button = widgets.Button(
            description="Save Selection",
            disabled=False,
            button_style="",
            tooltip="Click me",
            icon="",
        )
        self.best_n_button.on_click(self.save_best_of_SIRS)

        best_n_hbox = widgets.HBox()
        best_n_hbox.children += (self.best_n_checkbox,)
        best_n_hbox.children += (self.best_n_value,)
        best_n_hbox.children += (self.best_n_button,)

        self.cheb_checkbox = widgets.Checkbox(
            value=False,
            description="Fit Rational: ",
            disabled=False,
            indent=False,
            layout=widgets.Layout(width="180px", height="30px"),
        )
        self.cheb_value = widgets.Text(
            value="4",
            placeholder="",
            disabled=False,
            layout=widgets.Layout(width="50px", height="30px"),
        )
        self.cheb_value2 = widgets.Text(
            value="4",
            placeholder="",
            disabled=False,
            layout=widgets.Layout(width="50px", height="30px"),
        )

        cheb_hbox = widgets.HBox()
        cheb_hbox.children += (self.cheb_checkbox,)
        cheb_hbox.children += (self.cheb_value,)
        cheb_hbox.children += (self.cheb_value2,)

        active_file_hbox = widgets.HBox()
        active_file_hbox.children += (self.active_file,)
        active_file_hbox.children += (self.active_color,)
        active_file_hbox.children += (self.legend_checkbox,)
        active_file_hbox.children += (self.legend_str,)

        file_wildcard_hbox = widgets.HBox()
        file_wildcard_hbox.children += (
            widgets.Label("Select files (hold Ctrl for multiple)"),
        )
        file_wildcard_hbox.children += (self.wildcard_str,)

        file_vbox.children += (file_wildcard_hbox,)
        file_vbox.children += (self.file_select,)
        file_vbox.children += (active_file_hbox,)
        file_vbox.children += (color_fading_hbox,)
        file_vbox.children += (data_filtering_hbox,)
        file_vbox.children += (best_n_hbox,)
        file_vbox.children += (cheb_hbox,)
        file_vbox.children += (gpt_run_hbox,)

        self.x_select = widgets.Dropdown(
            options=["Temp"],
            value="Temp",
            description="x :",
            disabled=False,
            layout=widgets.Layout(width="250px", height="30px"),
        )
        self.y_select = widgets.Dropdown(
            options=["Temp"],
            value="Temp",
            description="y :",
            disabled=False,
            layout=widgets.Layout(width="250px", height="30px"),
        )
        self.c_select = widgets.Dropdown(
            options=["Temp"],
            value="Temp",
            description="color :",
            disabled=False,
            layout=widgets.Layout(width="250px", height="30px"),
        )

        scale_list = [str(xx) for xx in np.arange(-15, 16, 3)]
        self.x_scale = widgets.Dropdown(
            options=scale_list,
            value="0",
            description="scale : 10^",
            disabled=False,
            layout=widgets.Layout(width="140px", height="30px"),
        )
        self.x_units = widgets.Text(
            value="",
            placeholder="Enter units",
            disabled=False,
            layout=widgets.Layout(width="110px", height="30px"),
        )
        self.x_label = widgets.Text(
            value="",
            placeholder="Enter label",
            description="Label : ",
            disabled=False,
            layout=widgets.Layout(width="250px", height="30px"),
        )
        self.x_min = widgets.Text(
            value="",
            placeholder="x min",
            description="Limits : ",
            disabled=False,
            layout=widgets.Layout(width="150px", height="30px"),
        )
        self.x_max = widgets.Text(
            value="",
            placeholder="x max",
            disabled=False,
            layout=widgets.Layout(width="70px", height="30px"),
        )

        self.y_scale = widgets.Dropdown(
            options=scale_list,
            value="0",
            description="scale : 10^",
            disabled=False,
            layout=widgets.Layout(width="140px", height="30px"),
        )
        self.y_units = widgets.Text(
            value="",
            placeholder="Enter units",
            disabled=False,
            layout=widgets.Layout(width="110px", height="30px"),
        )
        self.y_label = widgets.Text(
            value="",
            placeholder="Enter label",
            description="Label : ",
            disabled=False,
            layout=widgets.Layout(width="250px", height="30px"),
        )
        self.y_min = widgets.Text(
            value="",
            placeholder="y min",
            description="Limits : ",
            disabled=False,
            layout=widgets.Layout(width="150px", height="30px"),
        )
        self.y_max = widgets.Text(
            value="",
            placeholder="y max",
            disabled=False,
            layout=widgets.Layout(width="70px", height="30px"),
        )

        self.c_scale = widgets.Dropdown(
            options=scale_list,
            value="0",
            description="scale : 10^",
            disabled=False,
            layout=widgets.Layout(width="140px", height="30px"),
        )
        self.c_units = widgets.Text(
            value="",
            placeholder="Enter units",
            disabled=False,
            layout=widgets.Layout(width="110px", height="30px"),
        )
        self.c_label = widgets.Text(
            value="",
            placeholder="Enter label",
            description="Label : ",
            disabled=False,
            layout=widgets.Layout(width="250px", height="30px"),
        )
        self.c_min = widgets.Text(
            value="",
            placeholder="c min",
            description="Limits : ",
            disabled=False,
            layout=widgets.Layout(width="150px", height="30px"),
        )
        self.c_max = widgets.Text(
            value="",
            placeholder="c max",
            disabled=False,
            layout=widgets.Layout(width="70px", height="30px"),
        )

        x_scale_hbox = widgets.HBox()
        x_scale_hbox.children += (self.x_scale,)
        x_scale_hbox.children += (self.x_units,)

        x_limits_hbox = widgets.HBox()
        x_limits_hbox.children += (self.x_min,)
        x_limits_hbox.children += (self.x_max,)

        y_scale_hbox = widgets.HBox()
        y_scale_hbox.children += (self.y_scale,)
        y_scale_hbox.children += (self.y_units,)

        y_limits_hbox = widgets.HBox()
        y_limits_hbox.children += (self.y_min,)
        y_limits_hbox.children += (self.y_max,)

        c_scale_hbox = widgets.HBox()
        c_scale_hbox.children += (self.c_scale,)
        c_scale_hbox.children += (self.c_units,)

        c_limits_hbox = widgets.HBox()
        c_limits_hbox.children += (self.c_min,)
        c_limits_hbox.children += (self.c_max,)

        var_select_vbox = widgets.VBox()
        var_select_vbox.children += (self.x_select,)
        var_select_vbox.children += (x_scale_hbox,)
        var_select_vbox.children += (self.x_label,)
        var_select_vbox.children += (x_limits_hbox,)
        var_select_vbox.children += (
            widgets.VBox(layout=widgets.Layout(width="20px", height="30px")),
        )
        var_select_vbox.children += (self.y_select,)
        var_select_vbox.children += (y_scale_hbox,)
        var_select_vbox.children += (self.y_label,)
        var_select_vbox.children += (y_limits_hbox,)
        var_select_vbox.children += (
            widgets.VBox(layout=widgets.Layout(width="20px", height="30px")),
        )
        var_select_vbox.children += (self.c_select,)
        var_select_vbox.children += (c_scale_hbox,)
        var_select_vbox.children += (self.c_label,)
        var_select_vbox.children += (c_limits_hbox,)

        input_hbox.children += (file_vbox,)
        input_hbox.children += (var_select_vbox,)

        # Settings region
        self.settings_box = widgets.Textarea(
            value="Click a point to see settings",
            placeholder="",
            description="",
            disabled=False,
            layout=widgets.Layout(width="1350px", height="150px"),
        )

        # Assemble GUI
        gui_top = widgets.HBox(layout={"border": "1px solid grey"})
        gui_top.children += (input_hbox,)
        gui_top.children += (fig_hbox,)

        self.gui.children += (gui_top,)
        self.gui.children += (self.settings_box,)

        self.pop_list = []
        self.settings = {}
        self.run_settings = {}
        self.ran_settings = {}

        self.dummy = 0

        self.make_gui()

    def make_gui(self):
        self.load_and_plot_on_value_change(None)
        display(self.gui)

        self.x_select.observe(self.reset_units_and_plot_on_value_change, names="value")
        self.y_select.observe(self.reset_units_and_plot_on_value_change, names="value")
        self.c_select.observe(self.reset_units_and_plot_on_value_change, names="value")
        self.file_select.observe(self.load_and_plot_on_value_change, names="value")
        self.x_scale.observe(self.plot_on_value_change, names="value")
        self.x_units.observe(self.plot_on_value_change, names="value")
        self.x_label.observe(self.plot_on_value_change, names="value")
        self.y_scale.observe(self.plot_on_value_change, names="value")
        self.y_units.observe(self.plot_on_value_change, names="value")
        self.y_label.observe(self.plot_on_value_change, names="value")
        self.c_scale.observe(self.plot_on_value_change, names="value")
        self.c_units.observe(self.plot_on_value_change, names="value")
        self.c_label.observe(self.plot_on_value_change, names="value")

        self.x_min.observe(self.plot_on_value_change, names="value")
        self.x_max.observe(self.plot_on_value_change, names="value")
        self.y_min.observe(self.plot_on_value_change, names="value")
        self.y_max.observe(self.plot_on_value_change, names="value")
        self.c_min.observe(self.plot_on_value_change, names="value")
        self.c_max.observe(self.plot_on_value_change, names="value")
        self.legend_checkbox.observe(self.plot_on_value_change, names="value")
        self.show_constraint_violators_checkbox.observe(
            self.plot_on_value_change, names="value"
        )

        self.active_file.observe(self.active_file_change, names="value")

        self.active_color.observe(self.active_color_change, names="value")
        self.legend_str.observe(self.legend_str_change, names="value")

        self.settings_menu.observe(self.show_settings, names="value")
        self.settings_value.observe(self.edit_settings_to_run, names="value")

        self.best_n_checkbox.observe(self.load_and_plot_on_value_change, names="value")
        self.best_n_value.observe(self.load_and_plot_on_value_change, names="value")

        self.cheb_checkbox.observe(self.plot_on_value_change, names="value")
        self.cheb_value.observe(self.plot_on_value_change, names="value")
        self.cheb_value2.observe(self.plot_on_value_change, names="value")

        self.color_fading_checkbox.observe(self.plot_on_value_change, names="value")
        self.color_fading_alpha.observe(self.plot_on_value_change, names="value")

        self.wildcard_str.observe(
            self.refresh_files_load_and_plot_on_value_change, names="value"
        )

    def on_click(self, event):
        which_line = self.snap_cursor.which_line
        which_point = self.snap_cursor.data_index

        pop = self.pop_list[which_line]
        if not self.show_constraint_violators_checkbox.value:
            pop = self.remove_constraint_violators(copy.copy(pop))

        pop_index = np.array(pop.index)

        all_settings = pop.to_dict("index")[pop_index[which_point]]

        pop_filename = os.path.join(
            self.pop_directory, self.file_select.value[which_line]
        )

        wanted_keys = list({**self.vocs.variables, **self.vocs.constants}.keys())
        if "merit:min_mean_z" in all_settings.keys():
            wanted_keys.append("merit:min_mean_z")

        self.settings = dict(
            (k, all_settings[k]) for k in wanted_keys if k in all_settings
        )
        self.run_settings = copy.copy(self.settings)
        self.settings_box.value = f"index = {pop_index[which_point]} in {pop_filename}\n{self.x_select.value} = {all_settings[self.x_select.value]:.7g}\n{self.y_select.value} = {all_settings[self.y_select.value]:.7g}\nsettings = {self.settings}"
        self.run_gpt_button.disabled = False
        self.settings_menu.disabled = False
        self.settings_value.disabled = False

        self.settings_menu.options = self.settings.keys()

        self.settings_value.unobserve_all(name="value")
        self.settings_value.value = str(self.settings[self.settings_menu.value])
        self.settings_value.observe(self.edit_settings_to_run, names="value")

    def reset_units(self, owner):
        if owner == self.x_select:
            self.x_scale.value = "0"
            self.x_units.value = ""
            self.x_label.value = ""

        if owner == self.y_select:
            self.y_scale.value = "0"
            self.y_units.value = ""
            self.y_label.value = ""

        if owner == self.c_select:
            self.c_scale.value = "0"
            self.c_units.value = ""
            self.c_label.value = ""

    def put_file_list_in_widgets(self):
        # Repopulate file selection lists
        file_list = self.make_file_list()

        self.active_file.unobserve_all(name="value")
        self.file_select.unobserve_all(name="value")

        self.active_file.index = 0
        self.file_select.index = [0]

        self.file_select.options = file_list
        self.file_select.value = [file_list[0]]

        self.active_file.options = [file_list[0]]
        self.active_file.value = file_list[0]

        self.file_select.observe(self.load_and_plot_on_value_change, names="value")
        self.active_file.observe(self.active_file_change, names="value")

    def make_file_list(self):
        file_list = glob.glob(os.path.join(self.pop_directory, self.wildcard_str.value))
        if len(file_list) == 0:
            file_list = glob.glob(os.path.join(self.pop_directory, "*.csv"))

        file_list.sort(key=lambda x: os.path.getmtime(x))
        file_list.reverse()

        file_list = [os.path.basename(ff) for ff in file_list]
        for ii, ff in enumerate(file_list):
            self.default_legend_dict[ff] = f"file {ii + 1}"
            self.legend_dict[ff] = ""

        return file_list

    # def pop_sampler(self, data, new_pop_size):
    #     xopt = Xopt.from_file(self.xopt_filename)
    #     xopt.strict = False
    #
    #     vocs = xopt.vocs
    #     # vocs.constraints = {}  # At some point this didn't seem to work, but now it does...
    #
    #     toolbox = cnsga_toolbox(vocs)
    #
    #     pop = pop_from_data(data, vocs)
    #
    #     pop = toolbox.select(pop, new_pop_size)
    #     index_list = np.array([int(p.index) for p in pop])
    #
    #     return data.loc[data.index[index_list]]

    def next_default_color(self):
        c = self.default_color_list[0]
        self.default_color_list = self.default_color_list[1:] + [c]
        return c

    def update_active_file_list(self):
        self.active_file.unobserve_all(name="value")

        self.active_file.index = 0
        self.active_file.options = self.file_select.value

        for file_key in self.active_file.options:
            if file_key not in self.color_dict:
                self.color_dict[file_key] = self.next_default_color()

        self.active_file.observe(self.active_file_change, names="value")

    def update_active_file_params(self):
        self.active_color.unobserve_all(name="value")
        self.legend_str.unobserve_all(name="value")

        file_key = self.active_file.value
        self.active_color.value = self.color_dict[file_key]
        self.legend_str.value = self.legend_dict[file_key]

        self.active_color.observe(self.active_color_change, names="value")
        self.legend_str.observe(self.legend_str_change, names="value")

    def remove_constraint_violators(self, pop_df):
        # Remove individuals that threw an error
        pop_df = pop_df[pop_df["xopt_error"] != True]

        if self.vocs.constraints:
            for c, v in self.vocs.constraints.items():
                bin_opr, bound = v[0], v[1]
                bound = float(bound)

                if bin_opr == "LESS_THAN":
                    pop_df = pop_df[pop_df[c] < bound]
                elif bin_opr == "GREATER_THAN":
                    pop_df = pop_df[pop_df[c] > bound]

        return pop_df

    # SIRS = Selected Individuals for ReSubmission
    def save_best_of_SIRS(self, click):
        pop_filenames = list(self.file_select.value)

        if self.best_n_checkbox.value:
            if isinstance(self.best_n_value.value, float):
                best_n = int(np.ceil(float(self.best_n_value.value)))
                for ii, pop in enumerate(self.pop_list):
                    if best_n == len(pop):
                        f = pop_filenames[ii].replace(
                            ".csv", "-" + str(best_n) + "_best_of_SIRS.csv"
                        )
                        pop.to_csv(
                            os.path.join(self.pop_directory, f),
                            index_label="xopt_index",
                        )

                        self.put_file_list_in_widgets()

                        self.load_and_plot_on_value_change(None)

    def load_files(self):
        pop_filenames = list(self.file_select.value)

        self.pop_list = []
        self.filtered_pop_list = []

        old_x = self.x_select.value
        old_y = self.y_select.value
        old_c = self.c_select.value

        for f in pop_filenames:
            print(os.path.join(self.pop_directory, f))
            new_pop = pd.read_csv(
                os.path.join(self.pop_directory, f), index_col="xopt_index"
            )

            # if (self.best_n_checkbox.value):
            #    if isfloat(self.best_n_value.value):
            #        best_n = int(np.ceil(float(self.best_n_value.value)))
            #        if ((best_n) > 0) and ((best_n) <= len(new_pop)):
            #            new_pop = self.pop_sampler(new_pop, best_n)
            self.pop_list += [new_pop]

        dropdown_items = self.params_from_xopt
        for pop in self.pop_list:
            dropdown_items += list(pop.columns[1:])

        dropdown_items = list(dict.fromkeys(dropdown_items))  # remove duplicates

        if old_x in dropdown_items:
            self.x_select.options = dropdown_items
            self.x_select.value = old_x
        else:
            self.x_select.index = 0
            self.x_select.options = dropdown_items
            self.x_select.value = dropdown_items[0]

        if old_x in dropdown_items:
            self.y_select.options = dropdown_items
            self.y_select.value = old_y
        else:
            self.y_select.index = 0
            self.y_select.options = dropdown_items
            self.y_select.value = dropdown_items[1]

        if old_c in dropdown_items:
            self.c_select.options = ["None"] + dropdown_items
            self.c_select.value = old_c
        else:
            self.c_select.index = 0
            self.c_select.options = ["None"] + dropdown_items
            self.c_select.value = "None"

    def rat_poly(self, x, n1, *ab):
        # n1 = 3

        a = list(ab[0][:n1])
        b = [0, 1] + list(ab[0][n1:])
        return np.polynomial.polynomial.polyval(
            x, a
        ) / np.polynomial.polynomial.polyval(x, b)

        # return a[0]/np.abs(x)**(a[1]**2)*(1 + np.polynomial.polynomial.polyval(x, b))/(1.0 + np.abs(a[2])*np.abs(x)**len(b))

    def make_plot(self):
        if self.colorbar_instance is not None:
            self.colorbar_instance.remove()
            self.colorbar_instance = None

        self.ax.cla()
        if self.c_select.value != "None":
            cmin = np.min(
                [
                    np.min(
                        pop[self.c_select.value] * 10 ** (-float(self.c_scale.value))
                    )
                    for pop in self.pop_list
                ]
            )
            cmax = np.max(
                [
                    np.max(
                        pop[self.c_select.value] * 10 ** (-float(self.c_scale.value))
                    )
                    for pop in self.pop_list
                ]
            )
            if cmin >= cmax * (1.0 - 1.0e-14):  # What were we thinking?!
                cmin = 0.9 * cmin
                cmax = 1.1 * cmax

            if len(self.c_min.value) > 0:
                cmin = float(self.c_min.value)
            if len(self.c_max.value) > 0:
                cmax = float(self.c_max.value)
        sc = []
        pl = []

        pop_filenames = list(self.file_select.value)

        for ii, pop in enumerate(self.pop_list):
            ii_backwards = len(self.pop_list) - ii
            pop_filename = pop_filenames[ii]

            if not self.show_constraint_violators_checkbox.value:
                pop = self.remove_constraint_violators(copy.copy(pop))

            x = pop[self.x_select.value] * 10 ** (-float(self.x_scale.value))
            y = pop[self.y_select.value] * 10 ** (-float(self.y_scale.value))

            if self.c_select.value != "None":
                c = pop[self.c_select.value] * 10 ** (-float(self.c_scale.value))

            not_nan = np.logical_not(np.isnan(x))

            if self.c_select.value != "None":
                line_handle = self.ax.scatter(
                    x[not_nan],
                    y[not_nan],
                    10,
                    c=c[not_nan],
                    vmin=cmin,
                    vmax=cmax,
                    cmap="jet",
                    marker=".",
                    zorder=ii_backwards,
                )
                sc.append(line_handle)

            else:
                legend_str = self.default_legend_dict[pop_filename]
                if len(self.legend_dict[pop_filename]) > 0:
                    legend_str = self.legend_dict[pop_filename]
                if self.color_fading_checkbox.value:
                    line_color = self.color_dict[pop_filenames[0]]
                    if ii == 0:
                        line_alpha = 1.0
                    else:
                        if isinstance(self.color_fading_alpha.value, float):
                            line_alpha = float(self.color_fading_alpha.value)
                            line_alpha = np.min([np.max([0.001, line_alpha]), 1.0])
                        else:
                            line_alpha = 0.001
                else:
                    line_color = self.color_dict[pop_filename]
                    line_alpha = 1.0
                (line_handle,) = self.ax.plot(
                    x[not_nan],
                    y[not_nan],
                    ".",
                    color=line_color,
                    label=legend_str,
                    zorder=ii_backwards,
                    alpha=line_alpha,
                )
                pl.append(line_handle)

            if self.cheb_checkbox.value == True:
                n1 = float(self.cheb_value.value)
                n2 = float(self.cheb_value2.value)

                if isinstance(n1, float):
                    n1_cheb = int(np.min([5, np.max([0, n1])]))
                else:
                    n1_cheb = int(0)

                if isinstance(n2, float):
                    n2_cheb = int(np.min([5, np.max([0, n2])]))
                else:
                    n2_cheb = int(0)

                x_scale = np.mean(np.abs(x[not_nan]))
                y_scale = np.mean(np.abs(y[not_nan]))

                p0a = 2 * (np.random.rand(n1_cheb) - 0.5)
                p0b = 2 * (np.random.rand(n2_cheb) - 0.5)
                # p0a = np.ones(n1_cheb)
                # p0b = np.ones(n2_cheb)

                # p0a = p0a * 2.0**(-np.arange(0, n1_cheb))
                # p0b = p0b * 2.0**(-np.arange(0, n2_cheb))

                p0 = list(p0a) + list(p0b)
                p0, cov = curve_fit(
                    lambda xx, *aa: self.rat_poly(xx, n1_cheb, aa),
                    x[not_nan] / x_scale,
                    y[not_nan] / y_scale,
                    p0=p0,
                )
                x_cfit = np.linspace(np.min(x[not_nan]), np.max(x[not_nan]), 300)

                (line_handle,) = self.ax.plot(
                    x_cfit,
                    y_scale * self.rat_poly(x_cfit / x_scale, n1_cheb, p0),
                    "-",
                    color=self.color_dict[pop_filename],
                    label="Fit",
                    zorder=ii_backwards,
                )

        if self.legend_checkbox.value:
            self.ax.legend()

        if len(pl) > 0:
            snap_cursor = SnappingCursor(self.fig, self.ax, pl)
            self.mouse_event_handler_1 = self.fig.canvas.mpl_connect(
                "motion_notify_event", snap_cursor.on_mouse_move
            )
            self.mouse_event_handler_2 = self.fig.canvas.mpl_connect(
                "button_press_event", self.on_click
            )
            self.snap_cursor = snap_cursor

        if len(sc) > 0:
            self.colorbar_instance = plt.colorbar(sc[-1], ax=self.ax)

            if len(self.c_label.value) == 0:
                clabel_str = self.c_select.value
            else:
                clabel_str = self.c_label.value

            if len(self.c_units.value) > 0:
                clabel_str += f" ({self.c_units.value})"
            self.colorbar_instance.set_label(clabel_str)

            if self.mouse_event_handler_1 is not None:
                self.fig.canvas.mpl_disconnect(self.mouse_event_handler_1)
            if self.mouse_event_handler_2 is not None:
                self.fig.canvas.mpl_disconnect(self.mouse_event_handler_2)
            self.snap_cursor = []

        if len(self.x_min.value) > 0:
            self.ax.set_xlim(left=float(self.x_min.value))
        if len(self.x_max.value) > 0:
            self.ax.set_xlim(right=float(self.x_max.value))

        if len(self.y_min.value) > 0:
            self.ax.set_ylim(bottom=float(self.y_min.value))
        if len(self.y_max.value) > 0:
            self.ax.set_ylim(top=float(self.y_max.value))

        if len(self.x_label.value) == 0:
            xlabel_str = self.x_select.value
        else:
            xlabel_str = self.x_label.value

        if len(self.x_units.value) > 0:
            xlabel_str += f" ({self.x_units.value})"

        if len(self.y_label.value) == 0:
            ylabel_str = self.y_select.value
        else:
            ylabel_str = self.y_label.value

        if len(self.y_units.value) > 0:
            ylabel_str += f" ({self.y_units.value})"

        self.ax.set_xlabel(xlabel_str)
        self.ax.set_ylabel(ylabel_str)

    def reset_units_and_plot_on_value_change(self, change):
        self.reset_units(change["owner"])
        self.make_plot()

    def plot_on_value_change(self, change):
        self.make_plot()

    def refresh_files_load_and_plot_on_value_change(self, change):
        self.put_file_list_in_widgets()
        self.load_and_plot_on_value_change(change)

    def load_and_plot_on_value_change(self, change):
        self.load_files()
        self.update_active_file_list()
        self.update_active_file_params()
        self.make_plot()

    def active_file_change(self, change):
        self.update_active_file_params()
        self.make_plot()

    def active_color_change(self, change):
        self.color_dict[self.active_file.value] = self.active_color.value
        self.make_plot()

    def legend_str_change(self, change):
        self.legend_dict[self.active_file.value] = self.legend_str.value
        self.make_plot()

    def run_gpt(self, click):
        self.settings_box.value = "Running..."
        self.settings_box.value = "Got to here."

    def show_settings(self, change):
        self.settings_value.unobserve_all(name="value")
        self.settings_value.value = str(self.run_settings[self.settings_menu.value])
        self.settings_value.observe(self.edit_settings_to_run, names="value")

    def edit_settings_to_run(self, change):
        if type(self.settings[self.settings_menu.value]) is not str:
            if isinstance(self.settings_value.value, float):
                self.run_settings[self.settings_menu.value] = float(
                    self.settings_value.value
                )
        else:
            self.run_settings[self.settings_menu.value] = self.settings_value.value

        self.settings_box.value = (
            f"Settings modified:\nrun_settings = {self.run_settings}"
        )
