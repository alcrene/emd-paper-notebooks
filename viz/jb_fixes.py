"""
Some fixes which are only relevant for Jupyter Book – i.e. they are used to
make the inlined documentation nicer, but don’t affect the simulator itself.
"""

def display_dataframe_with_math(df, raw=False):
    """
    Display a Pandas DataFrame in output cell, such that math is correctly
    displayed also when exported to Jupyter Book.

    This is a workaround for issue #1501: https://github.com/executablebooks/jupyter-book/issues/1501
    """
    import re
    from IPython.display import HTML
    
    html = df.to_html()    
    raw_html = re.sub(r"\$.*?\$", lambda m: convert_tex_to_html(m[0], raw=True), html)
    return raw_html if raw else HTML(raw_html)

def convert_tex_to_html(html, raw=False):
    """
    Given some HTML text which may contain dollar-delimited math,
    use MyST-Parser to convert any such math into pure HTML.
    For simple expressions, this results in HTML which display reasonably well
    without the need for external librairies like MathJax.

    .. Todo:: Find a way to optionally export MathJax nodes instead of plain HTML.
    """
    import io
    import os
    import re
    import pandas as pd
    from textwrap import dedent
    from tempfile import NamedTemporaryFile
    from contextlib import redirect_stdout
    from IPython.display import HTML
    from myst_parser.parsers.docutils_ import cli_html
    
    # Manually apply the MyST parser to convert $-$ into MathJax’s HTML code
    frontmatter="""
    ---
    myst:
      enable_extensions: [dollarmath, amsmath]
    ---
    """
    with NamedTemporaryFile('w', delete=False) as f:
        f.write(dedent(frontmatter).strip())
        f.write("\n\n")
        f.write(html)
    with redirect_stdout(io.StringIO()) as sf:
        cli_html([f.name])
    fullhtml = sf.getvalue()  # Returns a large-ish HTML with the full styling header
    os.remove(f.name)
    # Strip HTML headers to keep only the body with converted math
    m = re.search(r'<body>\n<div class="document">([\s\S]*)</div>\n</body>', fullhtml)
    raw_html = m[1].strip()
    # Special case: if we provided a snippet with no HTML markup at all, don’t wrap the result
    # in <p> tags
    if "\n" not in html and "<" not in html:
        m = re.match(r"<p>(.*)</p>", raw_html)
        if m:  # Match was a success: there are <p> tags we can remove
            raw_html = m[1]
    # Manually display the result as HTML
    return raw_html if raw else HTML(raw_html)