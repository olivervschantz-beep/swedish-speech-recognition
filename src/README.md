# In this SRC folder we will implement the machine learning portion of this project.

# DATA - A folder where the following actions are performed:
# - PyTorch dataset + dataloader
# - normalisation, pad/trim, feature extraction (MFCC, m.m. )
# - downlaoding of wav, resampling, etc.

# MODEL - Model that picks word "kräftskiva", "sju", "korsord"
# - pronunciation_scorer.py # modell/delmodell som bedömer "uttal korrekt/inte"
# - word_classifier.py

# TRANING - We train the model to recognize the correct pronunciation from the wrong one
# - train_word_classifier.py
# - train_pronunciation.py

# EVALUATION 
# - evaluate_word_classifier.py
# - evaluate_pronunciation.py

# INFERENCE - function that takes in a wav → word + grade