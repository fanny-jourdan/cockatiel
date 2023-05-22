import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from .occlusion_tools import extract_clauses
plt.style.use('seaborn')


def print_legend(colors, label_to_criterion):
  '''
  colors and label_to_criterion are two dictionnaries
  '''
  html = []
  for label_id in label_to_criterion.keys():
    html.append(f'<span style="background-color: {colors[label_id]} 0.5); padding: 1px 5px; border: solid 3px ; border-color: {colors[label_id]} 1); #EFEFEF">{label_to_criterion[label_id]} </span>')
  display(HTML("<div style='display: flex; width: 400px; flex-wrap: wrap'>" +  " ".join(html) + " </div>" ))
  display(HTML('<br><br>'))



def viz_concepts(text, explanation, colors, ignore_words = [], extract_fct = "clause"):
  '''
  text, string
  explanation, np.array, output of the occlusion function
  ignore_words, list, same as occlusion function
  extract_fct, "word", "clause" or "sentence", same as occlusion function
  colors, dictonnary, same as print_legend
  '''
  try:
    text = text.decode('utf-8')
  except:
    text = str(text)
  
  if extract_fct == "clause":
    words = extract_clauses(text, clause_type = None)
  else:
    words = word_tokenize(text)

  l_phi = np.array(explanation)

  phi_html = []

  p = 0 # pointer to get current color for the words (it does not color words that have no phi)
  for i in range(len(words)):
    if words[i] not in ignore_words:
      k = 0
      for j in range(len(l_phi)):
        if l_phi[k][p] < l_phi[j][p]:
          k = j
      
      if l_phi[k][p] > 0.2:
        phi_html.append(f'<span style="background-color: {colors[k]} {l_phi[k][p]}); padding: 1px 5px; border: solid 3px ; border-color: {colors[k]} 1); #EFEFEF">{words[i]}</span>')
        p+= 1
      else:
        phi_html.append(f'<span style="background-color: rgba(233,30,99,0);  padding: 1px 5px; border: solid 3px ; border-color:  rgba(233,30,99,0); #EFEFEF">{words[i]}</span>')
        p +=1
    else:
      phi_html.append(f'<span style="background-color: rgba(233,30,99,0);  padding: 1px 5px; border: solid 3px ; border-color:  rgba(233,30,99,0); #EFEFEF">{words[i]}</span>')
  display(HTML("<div style='display: flex; width: 400px; flex-wrap: wrap'>" +  " ".join(phi_html) + " </div>" ))
  display(HTML('<br><br>'))