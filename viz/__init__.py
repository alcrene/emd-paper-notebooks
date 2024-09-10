# + [markdown] tags=[]
# # Visualization utilities

# + tags=[]
import seaborn as sns
import holoviews as hv

# + tags=[]
import logging
import shelve
from types import SimpleNamespace
from addict import Dict
logger = logging.getLogger(__name__)

# + tags=[]
# Hack to import ..config: add the root 'notebooks' directory to the path
# (This makes accessible modules that would be accessible if viz was condensed to a single module.
# It also makes any import used in a notebook also work here.)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # .viz directory should still have priority

# + tags=[]
try:
    import myst_nb
    from config import config
except ImportError as e:
    logger.error(f"Extended `glue` will not work due to the following error: {e}")


# + [markdown] tags=[]
# ## Custom glue

# + tags=[]
# Submodule objects to expose at top level
from .colorscheme import ColorScheme
from . import jb_fixes

# ## Extension to `glue`

def glue(name: str, variable, display: bool=True,
         raw_html:bool|str=False, raw_latex:bool|str=False, raw_myst:bool|str=False) -> None:
    """Custom glue with additional support for {glue:raw} role.
    
    If `raw_html` and/or `raw_latex` is provided, then an additional {glue:raw}
    role is made available for MyST documents. This will insert the content
    of those arguments wrapped with either a ``{raw-html}`` or ``{raw-latex}``
    role so they only show up in the specified output.
    An additional argument `raw_myst` is provided, for values that should be 
    inserted as-is into the MyST source. This can be useful if only the LaTeX
    source should be wrapped with a ``{raw-*}`` role. (This is the case for example
    with numerical values that would be inserted into dollared equations.)
    The main feature is that this is done *before* any processing by MyST parser,
    so the content can be included just about anywhere, such as math and headings.
    The content themselves can also include things like math '$' markers or markdown.

    As a convenience, it is also possible to pass `raw_html=True`. In this case
    ``str(variable)`` is used for the raw string. The same goes for `raw_latex` and `raw_myst`.

    LIMITATION: When inserting with {glue:raw}, the notebook path is ignored.
    This means that glue identifiers used with {glue:raw} must be unique
    throughout the whole book.

    .. Note:: The result of specifying both `raw_html` and `raw_myst` is undefined.

    .. DEVNOTE:: In practice we ended up using `raw_latex` and `raw_myst` almost
       exclusively, and setting them to the same value (to be used inside dollar signs).
       So there is still room for improvement with regards to the `raw` options.

    """
    myst_nb.glue(name, variable, display)
    with shelve.open(str(config.paths.glue_shelf)) as db:
        if raw_myst : db[name + "__myst"]  = str(variable) if raw_myst  is True else raw_myst
        if raw_html : db[name + "__html"]  = str(variable) if raw_html  is True else raw_html
        if raw_latex: db[name + "__latex"] = (str(variable).replace("μ", "\\mu ")  # greek letter
                                                           .replace("µ", "\\mu ")  # micro prefix
                                                           .replace("–", "--")
                                                           .replace("–", "---")
                                              if raw_latex is True else raw_latex)


# + [markdown] tags=[]
# ## Plotting

# + tags=[]
dims = Dict(
    matplotlib = SimpleNamespace(
        λ  = hv.Dimension("λ", label="wavelength", unit=r"$\mu\mathrm{m}$"),
        B  = hv.Dimension("B", label="radiance", unit=r"$\mathrm{kW} \cdot \mathrm{sr}^{-1} \cdot \mathrm{m}^{-2} \cdot \mathrm{nm}^{-1}$"),
        t     = hv.Dimension("t", label="time", unit=r"$\mathrm{ms}$"),
        I_ext = hv.Dimension("I_ext", label="external input", unit=r"$\mathrm{nA}$"),
        Δt = hv.Dimension("Δt", label="time lag", unit=r"$\mathrm{ms}$"),
        I  = hv.Dimension("I", unit=r"$\mathrm{nA}$"),
        I2 = hv.Dimension("I2", label=r"$\langle I^2 \rangle$", unit=r"$\mathrm{nA}^2$"),
        V  = hv.Dimension("V", unit=r"$\mathrm{mV}$"),
        logL = hv.Dimension("logL", label="log likelihood"),
        Φ  = hv.Dimension("Φ", label=r"cum. prob. ($\Phi$)"),
        q  = hv.Dimension("q", label="loss ($q$)"),
        Δq  = hv.Dimension("Δq", label="loss increment ($\Delta q$)"),
        Bemd = hv.Dimension("Bemd", label=r"$B^{\mathrm{EMD}}$"),
        Bconf = hv.Dimension("Bconf", label=r"$B^{\mathrm{conf}}$"),
        c = hv.Dimension("c", label="$c$"),
        R = hv.Dimension("R", label="expected risk ($R$)")
    ),
    bokeh = SimpleNamespace(
        λ  = hv.Dimension("λ", label="wavelength", unit=r"$μm$"),
        B  = hv.Dimension("B", label="radiance", unit=r"$kW \cdot sr⁻¹ m⁻² nm⁻¹"),
        t     = hv.Dimension("t", label="time", unit="ms"),
        I_ext = hv.Dimension("I_ext", label="external input", unit="nA"),
        Δt = hv.Dimension("Δt", label="time lag", unit="ms"),
        I  = hv.Dimension("I", unit="nA"),
        I2 = hv.Dimension("I2", label="⟨I²⟩", unit="nA"),
        V  = hv.Dimension("V", unit="mV"),
        logL = hv.Dimension("logL", label="log likelihood"),
        Φ  = hv.Dimension("Φ", label="cum. prob. (Φ)"),
        q  = hv.Dimension("q", label="loss (q)"),
        Bemd = hv.Dimension("Bemd"),
        Bconf = hv.Dimension("Bconf"),
        c = hv.Dimension("c"),
        R = hv.Dimension("R", label="expected risk (R)")
    )
)


# -

def save(*args, **kwargs):
    """
    Wrapper for `holoviews.save` which allows to turn off
    saving plots globally: to turn off saving, assign `False` to:

        save.update_figure_files = False

    This can be useful to test running a notebook without risking to clobber
    the figure files. It also makes running the notebook faster.
    """
    if save.update_figure_files:
        hv.save(*args, **kwargs)
save.update_figure_files = True

# + [markdown] tags=[]
# ### Plot hooks

# + tags=[]
import matplotlib as mpl


# -

def noaxis_hook(plot, element):
    """Holoviews hook for plots using the matplotlib backend.
    Removes the axes.
    """
    plot.handles["axis"].set_axis_off()

def xaxis_off_hook(plot, element):
    plot.handles["axis"].get_xaxis().set_visible(False)
    no_spine_hook("top", "bottom")(plot, element)

def yaxis_off_hook(plot, element):
    plot.handles["axis"].get_yaxis().set_visible(False)
    no_spine_hook("left", "right")(plot, element) 

def no_spine_hook(*sides):
    """Remove the specified spine(s) completely. Matplotlib hook."""
    def hook(plot, element):
        ax = plot.handles["axis"]
        for side in sides:
            ax.spines[side].set_visible(False)
    return hook

def set_major_xticks_formatter(formatter):
    def hook(plot, element):
        plot.handles["axis"].get_xaxis().set_major_formatter(formatter)
    return hook
def set_xticklabels_hook(ticks, **kwargs):
    def hook(plot, element):
        plot.handles["axis"].get_xaxis().set_ticklabels(ticks, **kwargs)
    return hook
def set_yticklabels_hook(ticks, **kwargs):
    def hook(plot, element):
        plot.handles["axis"].get_yaxis().set_ticklabels(ticks, **kwargs)
    return hook

def set_xticks_hook(ticks, **kwargs):
    def hook(plot, element):
        plot.handles["axis"].get_xaxis().set_ticks(ticks, **kwargs)
    return hook
def set_yticks_hook(ticks, **kwargs):
    def hook(plot, element):
        plot.handles["axis"].get_yaxis().set_ticks(ticks, **kwargs)
    return hook

def no_xticks_hook(plot, element):
    plot.handles["axis"].get_xaxis().set_ticks([])
def no_yticks_hook(plot, element):
    plot.handles["axis"].get_yaxis().set_ticks([])

def hide_minor_ticks_hook(plot, element):
    ax = plot.handles["axis"]
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
def set_minor_xticks_formatter(formatter):
    def hook(plot, element):
        plot.handles["axis"].get_xaxis().set_minor_formatter(formatter)
    return hook

def despine_hook(offset=5, trim=True):
    def hook(plot, element, offset=offset, trim=trim, **kwargs):
        """Apply seaborn.despine. Matplotlib hook."""
        sns.despine(ax=plot.handles["axis"], **{"trim":trim, "offset":offset, **kwargs})
    if isinstance(offset, hv.plotting.Plot):
        # Not calling the function is equivalent to passing default args: offset, trim -> plot, element
        return hook(plot=offset, element=trim, offset=5, trim=True)
    else:
        return hook

def set_xlabel_hook(*args, **kwds):
    """Call `set_xlabel` with the provided arguments."""
    def hook(plot, element):
        ax = plot.handles["axis"]
        ax.set_xlabel(*args, **kwds)
    return hook
def set_ylabel_hook(*args, **kwds):
    """Call `set_ylabel` with the provided arguments."""
    def hook(plot, element):
        ax = plot.handles["axis"]
        ax.set_ylabel(*args, **kwds)
    return hook

def set_xlim_hook(*args, **kwargs):
    def hook(plot, element):
        plot.handles["axis"].set_xlim(*args, **kwargs)
    return hook
def set_ylim_hook(*args, **kwargs):
    def hook(plot, element):
        plot.handles["axis"].set_ylim(*args, **kwargs)
    return hook

# Based on https://stackoverflow.com/a/49449590
def xlabel_shift_hook(yshift=1, xshift=0):
    def hook(plot, element):
        ax = plot.handles["axis"]
        fig = plot.handles["fig"]
        shift = mpl.transforms.ScaledTranslation(xshift/72., yshift/72., fig.dpi_scale_trans)
        label = ax.xaxis.label
        label.set_transform(label.get_transform() + shift)
    return hook

# + tags=[]
def ylabel_shift_hook(xshift=1.5, yshift=0):
    def hook(plot, element):
        ax = plot.handles["axis"]
        fig = plot.handles["fig"]
        shift = mpl.transforms.ScaledTranslation(xshift/72., yshift/72., fig.dpi_scale_trans)
        label = ax.yaxis.label
        label.set_transform(label.get_transform() + shift)
    return hook

# + [markdown] tags=[]
# ## Plotting functions
# -

# ### `make_Rdist_fig`

def make_Rdist_fig(R_samples: dict, xticks: list, yticks: list, colors: hv.Cycle|list[str],
                   xticklabels=None, yticklabels=None, aspect=3):
    Rdists = [hv.Distribution(_Rlst, kdims=[dims.matplotlib.R], label=f"Model {a}")
              for a, _Rlst in R_samples.items()]
    Rcurves = [hv.operation.stats.univariate_kde(dist).to.curve()
               for dist in Rdists]
    fig_Rdists = hv.Overlay(Rdists) * hv.Overlay(Rcurves)

    # Plot styling
    if xticklabels is None and xticks and len(xticks) > 2:
        xticklabels = [str(xticks[0])] + [""]*(len(xticks)-2) + [str(xticks[-1])]
    if yticklabels is None and yticks and len(yticks) > 2:
        yticklabels = [str(yticks[0])] + [""]*(len(yticks)-2) + [str(yticks[-1])]
    hooks = []
    if xticklabels:
        hooks += [set_xticks_hook(xticks), set_xticklabels_hook(xticklabels), xlabel_shift_hook(7)]
    if yticklabels:
        hooks += [set_yticks_hook(yticks), set_yticklabels_hook(yticklabels), ylabel_shift_hook(5)]
    elif not yticks:  # yticks is empty or None: remove yaxis completely
        hooks += [yaxis_off_hook]  # NB: Using the Holoviews yaxis=None, which should have the same effect, doesn’t entirely work when other hooks are used.
    hooks += [despine_hook(2)]
    fig_Rdists.opts(
        hv.opts.Distribution(alpha=.3),
        hv.opts.Distribution(facecolor=colors, color="none", edgecolor="none", backend="matplotlib"),
        hv.opts.Curve(color=colors),
        hv.opts.Curve(linestyle="solid", backend="matplotlib"),
        hv.opts.Overlay(backend="matplotlib", fontscale=1.3,
                        hooks=hooks,
                        legend_position="top_left", legend_cols=1,
                        show_legend=False,
                        xlim=fig_Rdists.range("R"),  # Redundant, but ensures range is not changed
                        aspect=aspect
                       )
    )
    return fig_Rdists


# + [markdown] tags=[]
# ### `plot_param_space`
# Visualize a collection of parameters.
# Collection is packaged as a `ParamColl` instance, which defines certain parameters as random: by sampling these parameters, the new parameter sets can be created on demand.
# (This is similar to the [parameter space](https://pythonhosted.org/NeuroTools/parameters.html#the-parameterspace-class) idea of NeuroTools, which is now largely abandoned.)
# This function takes a `ParamColl` instance, samples all random parameters, and returns a `HoloMap` which can be used to visualize them.
#
# :::{note}
#
# `plot_param_space` is not currently used, but may still be useful as a generic function for other projects
# :::

# + [markdown] tags=[]
#     def _plot_param_hist_dict(Θcoll: ParamColl, θdim="param"):
#         rv_params = [name for name, val in Θcoll.items() if isinstance(val, ExpandableRV)]
#         kdims = {name: Θcoll.dims.get(name, name) for name in rv_params}
#         kdims = {name: dim if isinstance(dim, hv.Dimension) else hv.Dimension(name, label=dim) for name, dim in kdims.items()}
#         pdims = {name: hv.Dimension(f"p{name}", label=f"p({kdims[name].label})") for name in rv_params}
#         hists = {name: hv.Histogram(np.histogram(Θcoll[name].rvs(1000), bins="auto", density=True),
#                                     kdims=kdims[name], vdims=pdims[name])
#                  for name in rv_params}
#         # Augment with histograms from nested ParamColls
#         nested_paramcolls = [(name, val) for name, val in Θcoll.items() if isinstance(val, ParamColl)]
#         for name, paramcoll in nested_paramcolls:
#             new_hists = {f"{name}.{pname}": phist
#                          for pname, phist in _plot_param_hist_dict(paramcoll, θdim).items()}
#             hists = {**hists, **new_hists}
#         return hists

# + [markdown] tags=[]
#     def plot_param_space(Θcoll: ParamColl, θdim="param"):
#         hists = _plot_param_hist_dict(Θcoll, θdim)
#         return hv.HoloMap(hists, kdims=[θdim]).layout().cols(3)
# -

# ## Format numbers

# + tags=[]
import math
from numbers import Integral, Real
from typing import Literal
from TexSoup import TexSoup

# + tags=[]
def _int_str_to_superscript(v: str) -> str:
    exponents = list("⁰¹²³⁴⁵⁶⁷⁸⁹")
    assert set(v) <= set("0123456789")
    return "".join(exponents[ord(d)-ord("0")] for d in str(v))

# + tags=[]
def make_num_superscript(v: Real) -> str:
    """
    Convert a number to a string of superscript digits.
    Negative values are supported.
    """
    if isinstance(v, str):
        raise NotImplementedError
    s = []
    if v == 0:
        return "⁰"
    elif v < 0:
        s.append("⁻")
        v = -v
    if isinstance(v, Integral):
        s.append(_int_str_to_superscript(str(v)))
    elif isinstance(v, Real):
        i, d = str(v).split(".")  # Split into integer and decimal parts
        s.append(_int_str_to_superscript(str(i)))
        if not v.is_integer():
            s.append(".")
            s.append(_int_str_to_superscript(str(d)))
    return "".join(s)

# + tags=[]
def format_pow2(a: float, format:Literal["latex","$latex$","unicode"]="unicode") -> str:
    """Format a number as a power of 2.
    
    Technically works with any number, but mostly intended for powers of 2.
    The default returns a unicode string; pass ``format="latex"`` to get a
    format better suited for insertion into a string.

    TODO: Support "siunitx" format, which returns a quantity formatted with {:~Lx}
    """
    units = getattr(a, "units", "")
    if units:
        match format:
            case "unicode":
                units = f" {units:~P}"
            case "latex" | "$latex$" | "latex-with-dollar":
                units = f" {units:~L}"
            case _:
                raise ValueError(f"`format` argument must be one of 'unicode', 'latex' or '$latex$'. Received '{format}'.")
        a = a.magnitude
    p = math.log2(a)
    if p.is_integer(): p = int(p)
    if format in {"latex", "$latex$", "latex-with-dollar"}:
        s = f"2^{{{p}}}"
        if units:
            s += f"\\,\\mathrm{{{units}}}"  # TODO: Use \siunitx, as in display_quantity
        if format == "$latex$" or format == "latex-with-dollar":
            s = f"${s}$"
        return s
    elif format == "unicode":
        return f"2{make_num_superscript(p)}{units}"
    else:
        raise ValueError(f"`format` argument must be one of 'unicode', 'latex' or '$latex$'. Received '{format}'.")

def format_pow10(a: float, format:Literal["latex","latex-with-dollar","unicode"]="unicode") -> str:
    """Format a number as a power of 10.
    
    Technically works with any number, but mostly intended for powers of 10.
    The default returns a unicode string; pass ``format="latex"`` to get a
    format better suited for insertion into a string.
    """
    units = getattr(a, "units", "")
    if units:
        match format:
            case "unicode":
                units = f" {units:~P}"
            case "latex" | "latex-with-dollar":
                units = f" {units:~L}"
            case _:
                raise ValueError(f"`format` argument must be one of 'unicode', 'latex'. Received '{format}'.")
        a = a.magnitude
    p = math.log10(a)
    if p.is_integer(): p = int(p)
    if format.startswith("latex"):
        s = f"10^{{{p}}}"
        if units:
            s += f"\\,\\mathrm{{{units}}}"  # TODO: Use \siunitx, as in display_quantity
        if format == "latex-with-dollar":
            s = f"${s}$"
        return s
    elif format == "unicode":
        return f"10{make_num_superscript(p)}{units}"
    else:
        raise ValueError(f"`format` argument must be one of 'unicode', 'latex'. Received '{format}'.")

# + tags=[]
def format_scientific(a: int|float, sig_digits=3, min_power=1,
      format:Literal["latex","latex-with-dollar","unicode"]="unicode"
      ) -> str:
    """
    Format a number in scientific notation, with the requested number of
    significant digits.
    :param:min_power: Minimum power for scientific notation. If the power of the
        scientific notation would be less than this number, use decimal formatting.
        Set to ``0`` to force scientific notation for all values.
    :param:format: If "latex", return a LateX string without '$' markers.
        If "latex-with-dollar", return a LateX string with a '$' markers at each end.
        Otherwise return a Unicode string.
    """
    latex = (format.startswith("latex") or format == "siunitx")
    # First deal with the sign, since log10 requires strictly positive values
    if a < 0:
        sgn = "-"
        a = -a
    elif a == 0:
        return "0." + "0"*(sig_digits-1)
    else:
        sgn = ""
    # First round the number to avoid issues with things like 0.99999
    ## vvv EARLY EXITS vvv ##
    if not math.isfinite(a):
        if a == math.inf:
            s = "\\infty" if latex else "∞"
        elif a == -math.inf:
            s = "-\\infty" if latex else "-∞"
        else:
            return str(a)  # Never wrap with '$' markers
        if format in ("latex-with-dollar", "siunitx"):  # We don’t wrap numbers with siunitx, but we do wrap special symbols like \infty
            s = f"${s}$"
        return s
    ## ^^^ EARLY EXITS ^^^ ##
    p = int(math.log10(a))
    if p < 0 or (p == 0 and a < 1):
        a = round(a / 10**p, sig_digits)  # Need one digit more, because we use the first one to replace the initial '0.'
    else:
        a = round(a / 10**p, sig_digits-1)
        
    # Now we have a good value a with non-pathological loading on the first digit
    # Since a has changed though, we should check again that p is correct
    p2 = int(math.log10(a))  # Most of the time this will be 0
    p += p2
    i, f = str(float(a/10**p2)).split(".")
    if i == "0":
        i, f = f[0], f[1:]
        p -= 1
    f = (f+"0"*sig_digits)[:sig_digits-1]  # NB: This is not equivalent to {f:0<{sig_digits-1}} because it also truncates `f` if necessary
    #i = int(a // 10**p2)
    #f = str(a % 10**p2).replace('.','')[:sig_digits-1]
    sep = "." if f else ""
    if abs(p) < min_power:
        if p == 0:
            s = f"{sgn}{i}{sep}{f}"
        elif p < 0:
            s = f"{sgn}0.{'0'*(abs(p)-1)}{i}{f}"
        elif 0 < p < len(f):
            s = f"{sgn}{i}{f[:p]}{sep}{f[p:]}"
        else:
            assert p >= len(f), f"Unexpected power ('p') value: {p}"
            s = f"{sgn}{i}{f}{'0'*(p-len(f))}"
    else:
        if format.startswith("latex"):
            s = f"{sgn}{i}{sep}{f} \\times 10^{{{p}}}"
        elif format == "siunitx":
            s = f"{sgn}{i}{sep}{f}e{p}"
        else:
            s = f"{sgn}{i}{sep}{f}×10{make_num_superscript(p)}"

    if format == "latex-with-dollar":
        s = f"${s}$"
    return s

def tex_frac_to_solidus(texstr: str) -> str:
    """Convert a tex string with "\frac{a}{b}" into "{a}/{b}".
    The string must not contain anything else than the fraction
    """
    texsoup = TexSoup(f"${texstr}$".replace(")}$", ") }$"))
    if len(texsoup.contents) == 1 and len(texsoup[0].contents) == 1:
        cmd = texsoup[0][0]
        if cmd.name == "frac":
            assert len(cmd.args) == 2
            texstr = f"{cmd.args[0]} / {cmd.args[1]}"
    return texstr

def formatted_quantity(qty, uncertainty=None, precision=None) -> dict[str,str]:
    """Returns formatted strings for both HTML and LaTeX as a dict {'raw_html': str, 'raw_latex' str)
    Intended for passing to the `raw_html` and `raw_latex` argements of the extended `glue` function.
    """
    if precision is None and isinstance(uncertainty, int):
        logger.info("`precision` argument to `formatted_quantity` should be specified by keyword.")
        precision, uncertainty = uncertainty, precision
    Δ = uncertainty.to(qty.units) if uncertainty else None
    if precision is not None:
        qty_str = f"{qty.m:.{precision}f}"
    else:
        qty_str = f"{qty.m}"
    if uncertainty:
        htmlstr = f"({qty_str} \\pm {Δ.m:.{precision}f}) {qty.units:~L}"
        latexstr = f"\\qty{{{qty_str} \\pm {Δ.m:.{precision}f}}}{{{qty.units:~Lx}}}"
    else:
        htmlstr = f"{qty_str} {qty.units:~L}"
        latexstr = f"\\qty{{{qty_str}}}{{{qty.units:~Lx}}}"
    # Pint uses a macro with a weird unicode char for μm, which of course LaTeX is not happy with.
    # This isn’t even the char I get from typing 'μ': I needed to copy-paste from the output to get the test to work
    latexstr = latexstr.replace("\\µm", "\\micro\\metre")
    return {"variable": htmlstr, "raw_myst": htmlstr, "raw_latex": latexstr}

# + [markdown] tags=[]
# ### Test

# + tags=["active-ipynb"]
# if __name__ == "__main__":
#     assert format_scientific(0.9999999999999716) == "1.00"
#     assert format_scientific(1.0000000000000284) == "1.00"
#     assert format_scientific(9999.999999999716) == "1.00×10⁴"
#     assert format_scientific(1000.0000000000284) == "1.00×10³"
#     assert format_scientific(5.34) == "5.34"
#     assert format_scientific(32175254) == "3.22×10⁷"
#     assert format_scientific(0.000002789) == "2.79×10⁻⁶"
#     assert format_scientific(0.000002781) == "2.78×10⁻⁶"
#     
#     assert format_scientific(0.9999999999999716, sig_digits=1) == "1"
#     assert format_scientific(5.34, sig_digits=1) == "5"
#     assert format_scientific(5.74, sig_digits=1) == "6"
#     assert format_scientific(0.534, sig_digits=1) == "5×10⁻¹"
#     assert format_scientific(0.000002781, sig_digits=1) == "3×10⁻⁶"
#
#     assert format_scientific(0) == "0.00"
#
#     assert format_scientific(-0.9999999999999716) == "-1.00"
#     assert format_scientific(-1.0000000000000284) == "-1.00"
#     assert format_scientific(-9999.999999999716) == "-1.00×10⁴"
#     assert format_scientific(-1000.0000000000284) == "-1.00×10³"
#     assert format_scientific(-5.34) == "-5.34"
#     assert format_scientific(-32175254) == "-3.22×10⁷"
#     assert format_scientific(-0.000002789) == "-2.79×10⁻⁶"
#     assert format_scientific(-0.000002781) == "-2.78×10⁻⁶"
#
#     assert format_scientific(-0.9999999999999716, sig_digits=1) == "-1"
#     assert format_scientific(-5.34, sig_digits=1) == "-5"
#     assert format_scientific(-5.74, sig_digits=1) == "-6"
#     assert format_scientific(-0.534, sig_digits=1) == "-5×10⁻¹"
#     assert format_scientific(-0.000002781, sig_digits=1) == "-3×10⁻⁶"
