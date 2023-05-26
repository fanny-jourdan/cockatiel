"""
A module for generating plots for the explanations on notebooks.
"""
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import nltk
from nltk.tokenize import word_tokenize

from .utils import extract_clauses

from typing import List, Optional

nltk.download('punkt')
plt.style.use('seaborn')


def print_legend(colors, label_to_criterion):
    """
    Prints the legend for the plot in different colors.

    Parameters
    ----------
    colors
        A dictionary with the colors for each label.
    label_to_criterion
        A dictionary with the text to put on each label.
    """
    html = []
    for label_id in label_to_criterion.keys():
        html.append(f'<span style="background-color: {colors[label_id]} 0.5); padding: 1px 5px; border: solid 3px ; border-color: {colors[label_id]} 1); #EFEFEF">{label_to_criterion[label_id]} </span>')
    display(HTML("<div style='display: flex; width: 400px; flex-wrap: wrap'>" +  " ".join(html) + " </div>" ))
    display(HTML('<br><br>'))


def viz_concepts(
        text,
        explanation,
        colors,
        ignore_words: Optional[List[str]] = None,
        extract_fct: str = "clause"
):
    """
    Generates the visualization for COCKATIEL's explanations.

    Parameters
    ----------
    text
        A string with the text we wish to explain.
    explanation
        An array that corresponds to the output of the occlusion function.
    ignore_words
        A list of strings to ignore when applying occlusion.
    extract_fct
        A string indicating whether at which level we wish to explain: "word", "clause" or "sentence".
    colors
        A dictionary with the colors for each label
    """
    try:
        text = text.decode('utf-8')
    except:
        text = str(text)

    if extract_fct == "clause":
        words = extract_clauses(text, clause_type=None)
    else:
        words = word_tokenize(text)

    l_phi = np.array(explanation)

    phi_html = []

    p = 0  # pointer to get current color for the words (it does not color words that have no phi)
    for i in range(len(words)):
        if words[i] not in ignore_words:
            k = 0
            for j in range(len(l_phi)):
                if l_phi[k][p] < l_phi[j][p]:
                    k = j

            if l_phi[k][p] > 0.2:
                phi_html.append(f'<span style="background-color: {colors[k]} {l_phi[k][p]}); padding: 1px 5px; border: solid 3px ; border-color: {colors[k]} 1); #EFEFEF">{words[i]}</span>')
                p += 1
            else:
                phi_html.append(f'<span style="background-color: rgba(233,30,99,0);  padding: 1px 5px; border: solid 3px ; border-color:  rgba(233,30,99,0); #EFEFEF">{words[i]}</span>')
                p += 1
        else:
            phi_html.append(f'<span style="background-color: rgba(233,30,99,0);  padding: 1px 5px; border: solid 3px ; border-color:  rgba(233,30,99,0); #EFEFEF">{words[i]}</span>')
    display(HTML("<div style='display: flex; width: 400px; flex-wrap: wrap'>" +  " ".join(phi_html) + " </div>" ))
    display(HTML('<br><br>'))
