import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np

from .occlusion_tools import calculate_u_values, calculate_importance, extract_clauses


def occlusion_concepts(sentence, model, tokenizer, factorization, l_concept_id, ignore_words = [], two_labels = True, extract_fct = "clause", device = 'cuda'):
  '''
  sentence, a string
  model, the model explained by COCKATIEL
  tokenizer, the tokenizer associated at the model
  factorisation, a sklearn.decomposition._nmf.NMF if two_labels = False, a list of two sklearn.decomposition._nmf.NMF if two_labels = True
  l_concept_id, a np.ndarray if two_labels = False, a list of two np.ndarray if two_labels = True
  ignore_words, a list of words or characters on which we do not want to apply occlusion
  two_labels, a boolean, if two_labels = false, we look only one class, else, we look the both class of IMDB task
  extract_fct in ["word", "clause", "sentence"], the type of excerpts chosen for occlusion
  device : cuda or cpu, the device of the model
  '''
  sentence = str(sentence)

  if extract_fct == "clause":
    words = extract_clauses(sentence, clause_type = None)
    separate = " "

  else:
    words = word_tokenize(sentence)
    if extract_fct == "sentence":
      separate = ". "
    elif extract_fct == "word":
      separate = " "
    else:
      return("error, extract_fct can be only 'clause', 'sentence', or 'word")

  if two_labels :
      u_values_pos = calculate_u_values(sentence, words,  model, tokenizer, factorization[0], separate, ignore_words, device) 
      u_values_neg = calculate_u_values(sentence, words,  model, tokenizer, factorization[1], separate, ignore_words, device) 
      l_importances = []
      for concept_id in l_concept_id[0]:
        importances = calculate_importance(words, u_values_pos, concept_id, ignore_words)
        l_importances.append(np.array(importances))
      for concept_id in l_concept_id[1]:
        importances = calculate_importance(words, u_values_neg, concept_id, ignore_words)
        l_importances.append(np.array(importances))

  else: #look at only one class:
    u_values = calculate_u_values(sentence, words,  model, tokenizer, factorization, separate, ignore_words, device) 
    l_importances = []
    for concept_id in l_concept_id :
      importances = calculate_importance(words, u_values, concept_id, ignore_words)
      l_importances.append(np.array(importances))
    
  return(np.array(l_importances))


