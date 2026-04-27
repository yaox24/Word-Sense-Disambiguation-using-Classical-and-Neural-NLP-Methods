'''
Methods:
  1. Most Frequent Sense        (baseline)
  2. NLTK Lesk                  (baseline, with cleaned/preprocessed context)
  3. GloVe + MLP, unsupervised  (neural method 1 - pretrained embeddings)
  4. GloVe + MLP, supervised    (neural method 2 - trained on dev labels)
'''
import os
import numpy as np

# NLTK imports
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

# Download required NLTK data if not already present
for pkg in ['wordnet', 'stopwords', 'punkt', 'punkt_tab', 'omw-1.4']:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

from loader import load_instances, load_key


# =============================================================================
# SHARED HELPERS
# =============================================================================

STOP_WORDS = set(stopwords.words('english'))


def to_str(s):
    """Decode bytes to string (the loader may return byte strings)."""
    if isinstance(s, bytes):
        return s.decode('utf-8', errors='ignore')
    return s


def get_lemma_str(inst):
    """
    Return the lemma as a clean lowercase string.
    Multi-word lemmas like 'latin_america' keep their underscore so WordNet
    can look them up correctly (wn.synsets uses underscores, not spaces).
    """
    return to_str(inst.lemma).lower()   # e.g. 'latin_america' or 'bank'


def clean_context(context):
    """
    Turn the sentence context (a list of lemma strings/bytes) into a clean
    list of individual words, with stop words removed.

    Multi-word lemmas are split into their component
    words here because they appear as context tokens, not as lookup targets.
    We still want 'latin' and 'america' to contribute to the context vector.

    same function is used consistently by ALL methods (including Lesk).
    """
    words = []
    for token in context:
        token = to_str(token).lower()
        # split on underscore so multi-word tokens become individual words
        for w in token.replace('_', ' ').split():
            if w.isalpha() and w not in STOP_WORDS:
                words.append(w)
    return words


def get_synset_text_words(synset):
    """
    Collect all meaningful words from a synset's definition, examples,
    and lemma names. Used by both neural methods.
    """
    words = []
    # definition
    words += word_tokenize(synset.definition())
    # usage examples
    for ex in synset.examples():
        words += word_tokenize(ex)
    # lemma names (split multi-word ones, e.g. 'domestic_dog' -> 'domestic', 'dog')
    for lemma in synset.lemmas():
        words += lemma.name().lower().replace('_', ' ').split()
    # one level of hypernyms for extra context
    for hyp in synset.hypernyms():
        words += word_tokenize(hyp.definition())
    # keep only clean, non-stopword alphabetic tokens
    return [w.lower() for w in words if w.isalpha() and w not in STOP_WORDS]


def sense_key_to_synset(sense_key):
    """Convert a WordNet sense key string to a Synset object (or None)."""
    try:
        return wn.lemma_from_key(sense_key).synset()
    except Exception:
        return None


def evaluate(predictions, gold_key):
    """
    Compute accuracy.
    A prediction is correct if it matches ANY of the gold synsets for that instance.
    """
    correct = 0
    total = 0
    for inst_id, pred_synset in predictions.items():
        if inst_id not in gold_key:
            continue
        total += 1
        if pred_synset is None:
            continue
        gold_synsets = {sense_key_to_synset(sk) for sk in gold_key[inst_id]}
        gold_synsets.discard(None)
        if pred_synset in gold_synsets:
            correct += 1
    return correct / total if total > 0 else 0.0


# =============================================================================
# METHOD 1: MOST FREQUENT SENSE BASELINE
# =============================================================================

def most_frequent_sense(instances):
    """
    Always pick the first synset WordNet lists for the target lemma.
    WordNet orders synsets by corpus frequency, so synset #1 is the most common.
    This is a strong baseline that requires zero context.
    """
    predictions = {}
    for inst_id, inst in instances.items():
        lemma_str = get_lemma_str(inst)          # e.g. 'bank' or 'latin_america'
        synsets = wn.synsets(lemma_str)
        predictions[inst_id] = synsets[0] if synsets else None
    return predictions


# =============================================================================
# METHOD 2: NLTK LESK BASELINE (with cleaned/preprocessed context)
# =============================================================================

def nltk_lesk(instances):
    """
    Use NLTK's built-in Lesk algorithm.
    picks the synset whose definition shares the most words with the sentence.

    """
    predictions = {}
    for inst_id, inst in instances.items():
        lemma_str = get_lemma_str(inst)
        # lesk wants the ambiguous word without underscores
        ambiguous_word = lemma_str.replace('_', ' ')
        # Use the shared clean_context() for consistent preprocessing
        context_words = clean_context(inst.context)
        result = lesk(context_words, ambiguous_word)
        predictions[inst_id] = result
    return predictions


# =============================================================================
# GLOVE LOADING (shared by Methods 3 and 4)
# =============================================================================

def load_glove(glove_path):
    """
    Load GloVe vectors from a .txt file.
    Returns a dict:  word (str) -> numpy array of floats
    """
    print(f"Loading GloVe vectors from {glove_path} ...")
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            # glove.6B.100d.txt should have exactly 101 parts per line
            if len(parts) != 101:
                continue

            word = parts[0]

            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            embeddings[word] = vec
    print(f"  Loaded {len(embeddings)} word vectors "
          f"(dim={len(next(iter(embeddings.values())))})")
    return embeddings


def avg_vector(words, embeddings, dim):
    """
    Average the GloVe vectors for a list of words.
    Words not in the vocabulary are skipped.
    Returns a zero vector if no words are found.
    """
    vecs = [embeddings[w] for w in words if w in embeddings]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def cosine_sim(a, b):
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
# METHOD 3: GLOVE + MLP, UNSUPERVISED  (neural method 1)
# =============================================================================
#
# This is a proper neural network (MLP) that uses GloVe pretrained embeddings.
#
# Architecture (unsupervised - no gold labels needed):
#   Input:  [ctx_vec ; sense_vec ; cos_sim * ones(dim)]    (3 * dim features)
#   Hidden: one layer (dim=32) with ReLU
#   Output: a single score
#
# Key idea (unsupervised training):
#   no labelled training data for this method. 
#   Instead, use the GloVe vectors themselves as a self-supervised signal:
#     - Positive pair:  the target word vector vs. words in its synset gloss
#     - Negative pairs: same target word vs. words in randomly sampled glosses
#   The MLP learns a scoring function *beyond* raw cosine similarity.
#   At test time, the trained MLP scores each candidate synset and we pick the best.
#
# This differs from Method 4 in:
#   - Training signal (self-supervised GloVe pairs vs. gold labels)
#   - Feature interaction (scaled cosine vs. difference vector)
#   - Network size (hidden_dim=32 vs. 64)
#   - Loss function (MSE vs. cross-entropy)
# =============================================================================

class UnsupervisedMLP:
    """
    Small two-layer MLP, trained without gold WSD labels.
    Input -> Hidden (ReLU, dim=32) -> Output (scalar score)
    """

    def __init__(self, input_dim, hidden_dim=32, lr=0.005):
        # Xavier initialisation
        self.W1 = (np.random.randn(input_dim, hidden_dim).astype(np.float32)
                   * np.sqrt(2.0 / input_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (np.random.randn(hidden_dim, 1).astype(np.float32)
                   * np.sqrt(2.0 / hidden_dim))
        self.b2 = np.zeros(1, dtype=np.float32)
        self.lr = lr

    def forward(self, x):
        
        z1 = x @ self.W1 + self.b1         # (hidden_dim,)
        h  = np.maximum(0, z1)             # ReLU
        z2 = h @ self.W2 + self.b2         # (1,)
        return float(z2[0]), h, z1

    def train_step_mse(self, x, label):
        #One MSE gradient-descent update. Returns the MSE loss
        score, h, z1 = self.forward(x)
        # --- backward pass ---
        d_out  = score - label
        dW2    = h[:, None] * d_out
        db2    = np.array([d_out], dtype=np.float32)
        d_h    = self.W2[:, 0] * d_out
        d_z1   = d_h * (z1 > 0).astype(np.float32)
        dW1    = x[:, None] * d_z1[None, :]
        db1    = d_z1
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return (score - label) ** 2


def make_feature_m3(ctx_vec, sense_vec, dim):
    """
    Feature for Method 3.
    Concatenates -> context vector | sense vector | cosine_sim * ones(dim)

    The cosine similarity is broadcast as a constant vector so the MLP can
    learn a non-linear transformation of that similarity signal.
    """
    sim = cosine_sim(ctx_vec, sense_vec)
    sim_vec = np.full(dim, sim, dtype=np.float32)
    return np.concatenate([ctx_vec, sense_vec, sim_vec])


def glove_mlp_wsd(instances, embeddings, dim, hidden_dim=32, epochs=3, lr=0.005):
    """
    Neural method 1: unsupervised GloVe + MLP.

    Self-supervised training:
      For every (word, synset) pair that GloVe knows about, we create:
        - a positive training example (label = 1): the word vector paired with
          the average of the synset's gloss words
        - a negative training example (label = 0): the same word vector paired
          with a randomly chosen different synset's gloss words
      The MLP is trained with MSE loss to score positive pairs high and
      negative pairs low, purely from GloVe geometry.
    """
    np.random.seed(42)
    input_dim = dim * 3    # ctx + sense + cosine broadcast
    mlp = UnsupervisedMLP(input_dim, hidden_dim=hidden_dim, lr=lr)

   
    # Build self-supervised training examples from WordNet + GloVe.
    # use WordNet synsets whose main lemma exists in GloVe vocabulary.
   
    print("  Building self-supervised training pairs for Method 3 MLP...")
    all_synsets = list(wn.all_synsets())



    # Pre-compute sense vectors for all synsets (reused across epochs)
    sense_vecs = {}

    for syn in all_synsets:
        words = get_synset_text_words(syn)
        sense_vecs[syn] = avg_vector(words, embeddings, dim)


    # Build (word_vec, sense_vec, label) triples
    train_pairs = []
    rng = np.random.default_rng(42)

    for syn in all_synsets:
        # Represent this synset's "word" by the average of its lemma vectors
        lemma_words = [l.name().lower().replace('_', ' ').split()[0]
                       for l in syn.lemmas()]
        word_vec = avg_vector(lemma_words, embeddings, dim)

        if np.linalg.norm(word_vec) == 0:
            continue   # skip if no GloVe coverage

        pos_vec = sense_vecs[syn]

        # Positive pair
        feat_pos = make_feature_m3(word_vec, pos_vec, dim)
        train_pairs.append((feat_pos, 1.0))

        # One random negative pair
        neg_syn = all_synsets[rng.integers(len(all_synsets))]
        feat_neg = make_feature_m3(word_vec, sense_vecs[neg_syn], dim)
        train_pairs.append((feat_neg, 0.0))

    print(f"  Total training pairs: {len(train_pairs)}")

    # Train
    for epoch in range(epochs):
        idx = rng.permutation(len(train_pairs))
        total_loss = 0.0
        for i in idx:
            feat, label = train_pairs[i]
            total_loss += mlp.train_step_mse(feat, label)
        print(f"  Epoch {epoch+1}/{epochs}  avg loss: {total_loss/len(train_pairs):.4f}")

    # Predict
    predictions = {}
    for inst_id, inst in instances.items():
        lemma_str     = get_lemma_str(inst)
        context_words = clean_context(inst.context)
        ctx_vec       = avg_vector(context_words, embeddings, dim)

        synsets = wn.synsets(lemma_str)
        if not synsets:
            predictions[inst_id] = None
            continue

        best_synset = None
        best_score  = -1e9

        for syn in synsets:
            sense_vec = sense_vecs.get(syn)
            if sense_vec is None:
                sense_words = get_synset_text_words(syn)
                sense_vec   = avg_vector(sense_words, embeddings, dim)
            feat  = make_feature_m3(ctx_vec, sense_vec, dim)
            score, _, _ = mlp.forward(feat)
            if score > best_score:
                best_score  = score
                best_synset = syn

        predictions[inst_id] = best_synset

    return predictions


# =============================================================================
# METHOD 4: GLOVE + MLP, SUPERVISED  (neural method 2)
# =============================================================================
#
# Architecture:
#   Input:  [ctx_vec ; sense_vec ; ctx_vec - sense_vec]   (3 * dim features)
#   Hidden: one layer (dim=64) with ReLU
#   Output: sigmoid score (probability that this sense is correct)
#
# Training (supervised):
#   We train on the DEV set gold labels.  For each dev instance, the correct
#   synset(s) get label 1; all other candidates get label 0.
#   We use binary cross-entropy loss (different from Method 3's MSE).
#
# Method 3 vs Method 4 summary:
#   - Method 3: unsupervised, MSE loss, cosine-broadcast feature, hidden_dim=32
#   - Method 4: supervised (gold labels), BCE loss, difference feature, hidden_dim=64


class SupervisedMLP:
    """
    Two-layer feedforward network trained with binary cross-entropy loss.
    Input -> Hidden (ReLU, dim=64) -> Sigmoid output (0..1)
    """

    def __init__(self, input_dim, hidden_dim=64, lr=0.005):
        self.W1 = (np.random.randn(input_dim, hidden_dim).astype(np.float32)
                   * np.sqrt(2.0 / input_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (np.random.randn(hidden_dim, 1).astype(np.float32)
                   * np.sqrt(2.0 / hidden_dim))
        self.b2 = np.zeros(1, dtype=np.float32)
        self.lr = lr

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def forward(self, x):
        
        z1   = x @ self.W1 + self.b1     # (hidden_dim,)
        h    = np.maximum(0, z1)          # ReLU
        z2   = float((h @ self.W2 + self.b2)[0])
        prob = float(self.sigmoid(z2))
        return prob, z2, h, z1

    def train_step_bce(self, x, label):
        """
        One binary cross-entropy gradient-descent step.
        BCE loss: -[y*log(p) + (1-y)*log(1-p)]
        Returns the loss value.
        """
        prob, z2, h, z1 = self.forward(x)
        # Gradient of BCE w.r.t. sigmoid output is simply (prob - label)
        d_out = prob - label                           # scalar
        dW2   = h[:, None] * d_out
        db2   = np.array([d_out], dtype=np.float32)
        d_h   = self.W2[:, 0] * d_out
        d_z1  = d_h * (z1 > 0).astype(np.float32)
        dW1   = x[:, None] * d_z1[None, :]
        db1   = d_z1
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        # BCE loss
        eps  = 1e-9
        loss = -(label * np.log(prob + eps) + (1 - label) * np.log(1 - prob + eps))
        return float(loss)


def make_feature_m4(ctx_vec, sense_vec):
    """
    Feature for Method 4.
    Concatenates: context vector | sense vector | (context - sense) difference

    The difference vector encodes which dimensions the context and sense
    disagree on, giving the MLP a different inductive bias from Method 3.
    """
    diff = ctx_vec - sense_vec
    return np.concatenate([ctx_vec, sense_vec, diff]).astype(np.float32)


def supervised_mlp_wsd(train_instances, train_key, predict_instances,
                       embeddings, dim, hidden_dim=64, epochs=5, lr=0.005):
    """
    Neural method 2: supervised GloVe + MLP (binary cross-entropy).

    Training:
      For each dev instance, pair the context vector with each candidate synset
      vector. The gold synset(s) get label 1, all others get label 0.
      We shuffle training pairs each epoch.

    Prediction:
      Score all candidate synsets and pick the highest-probability one.
    """
    np.random.seed(42)
    input_dim = dim * 3            # ctx + sense + difference
    mlp = SupervisedMLP(input_dim, hidden_dim=hidden_dim, lr=lr)

    # Build training pairs
    print("  Building supervised training pairs for Method 4 MLP...")
    train_pairs = []   # list of (feature_vector, label)

    for inst_id, inst in train_instances.items():
        if inst_id not in train_key:
            continue

        lemma_str     = get_lemma_str(inst)
        context_words = clean_context(inst.context)
        ctx_vec       = avg_vector(context_words, embeddings, dim)

        synsets = wn.synsets(lemma_str)
        if not synsets:
            continue

        gold_synsets = {sense_key_to_synset(sk) for sk in train_key[inst_id]}
        gold_synsets.discard(None)

        for syn in synsets:
            sense_words = get_synset_text_words(syn)
            sense_vec   = avg_vector(sense_words, embeddings, dim)
            feat        = make_feature_m4(ctx_vec, sense_vec)
            label       = 1.0 if syn in gold_synsets else 0.0
            train_pairs.append((feat, label))

    print(f"  Total training pairs: {len(train_pairs)}")

    # Train
    rng = np.random.default_rng(42)
    for epoch in range(epochs):
        idx = rng.permutation(len(train_pairs))
        total_loss = 0.0
        for i in idx:
            feat, label = train_pairs[i]
            total_loss += mlp.train_step_bce(feat, label)
        print(f"  Epoch {epoch+1}/{epochs}  avg loss: {total_loss/len(train_pairs):.4f}")

    # Predict
    predictions = {}

    for inst_id, inst in predict_instances.items():
        lemma_str     = get_lemma_str(inst)
        context_words = clean_context(inst.context)
        ctx_vec       = avg_vector(context_words, embeddings, dim)

        synsets = wn.synsets(lemma_str)
        if not synsets:
            predictions[inst_id] = None
            continue

        best_synset = None
        best_score  = -1e9

        for syn in synsets:
            sense_words = get_synset_text_words(syn)
            sense_vec   = avg_vector(sense_words, embeddings, dim)
            feat        = make_feature_m4(ctx_vec, sense_vec)
            prob, _, _, _ = mlp.forward(feat)
            if prob > best_score:
                best_score  = prob
                best_synset = syn

        predictions[inst_id] = best_synset

    return predictions, mlp


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # File paths
    DATA_FILE  = 'multilingual-all-words.en.xml'
    KEY_FILE   = 'wordnet.en.key'
    GLOVE_FILE = 'glove.6B.100d.txt'  
    GLOVE_DIM  = 100

    # Load instances and gold keys
    print("Loading data...")
    dev_instances,  test_instances  = load_instances(DATA_FILE)
    dev_key,        test_key        = load_key(KEY_FILE)

    # Keep only instances that have a gold answer (as in the starter code)
    dev_instances  = {k: v for k, v in dev_instances.items()  if k in dev_key}
    test_instances = {k: v for k, v in test_instances.items() if k in test_key}

    print(f"Dev  instances: {len(dev_instances)}")
    print(f"Test instances: {len(test_instances)}")
    print()

    # Method 1: Most Frequent Sense 
    print("=" * 55)
    print("Method 1: Most Frequent Sense (baseline)")
    mfs_dev_preds  = most_frequent_sense(dev_instances)
    mfs_test_preds = most_frequent_sense(test_instances)
    mfs_dev_acc    = evaluate(mfs_dev_preds,  dev_key)
    mfs_test_acc   = evaluate(mfs_test_preds, test_key)
    print(f"  Dev  accuracy: {mfs_dev_acc:.4f}")
    print(f"  Test accuracy: {mfs_test_acc:.4f}")
    print()

    # Method 2: NLTK Lesk (with clean context)
    print("Method 2: NLTK Lesk (baseline, preprocessed context)")
    lesk_dev_preds  = nltk_lesk(dev_instances)
    lesk_test_preds = nltk_lesk(test_instances)
    lesk_dev_acc    = evaluate(lesk_dev_preds,  dev_key)
    lesk_test_acc   = evaluate(lesk_test_preds, test_key)
    print(f"  Dev  accuracy: {lesk_dev_acc:.4f}")
    print(f"  Test accuracy: {lesk_test_acc:.4f}")
    print()

    # Load GloVe (needed for methods 3 and 4)
    if not os.path.exists(GLOVE_FILE):
        print(f"ERROR: GloVe file not found at '{GLOVE_FILE}'.")
        print("Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/")
        print("and place glove.6B.100d.txt in the same folder as this script.")
        exit(1)

    embeddings = load_glove(GLOVE_FILE)
    print()

    # Method 3: GloVe + MLP, unsupervised 
    print("Method 3: GloVe + MLP, Unsupervised (neural method 1)")
    print("  (self-supervised training on GloVe pairs, MSE loss, hidden_dim=32)")
    glove_mlp_dev_preds  = glove_mlp_wsd(dev_instances,  embeddings, GLOVE_DIM)
    glove_mlp_test_preds = glove_mlp_wsd(test_instances, embeddings, GLOVE_DIM)
    glove_mlp_dev_acc    = evaluate(glove_mlp_dev_preds,  dev_key)
    glove_mlp_test_acc   = evaluate(glove_mlp_test_preds, test_key)
    print(f"  Dev  accuracy: {glove_mlp_dev_acc:.4f}")
    print(f"  Test accuracy: {glove_mlp_test_acc:.4f}")
    print()

    # Method 4: GloVe + MLP, supervised (train on dev, predict on test) 
    print("Method 4: GloVe + MLP, Supervised (neural method 2)")
    print("  (supervised on dev gold labels, BCE loss, hidden_dim=64)")
    print("  Training on dev set, predicting on test set...")
    mlp_test_preds, trained_mlp = supervised_mlp_wsd(
        train_instances   = dev_instances,
        train_key         = dev_key,
        predict_instances = test_instances,
        embeddings        = embeddings,
        dim               = GLOVE_DIM,
        hidden_dim        = 64,
        epochs            = 5,
        lr                = 0.005,
    )

    mlp_test_acc = evaluate(mlp_test_preds, test_key)
    print(f"  Test accuracy: {mlp_test_acc:.4f}")

    # Also score dev instances with the trained model (in-sample)
    print("  Predicting on dev set...")
    mlp_dev_preds = {}
    for inst_id, inst in dev_instances.items():

        lemma_str     = get_lemma_str(inst)
        context_words = clean_context(inst.context)
        ctx_vec       = avg_vector(context_words, embeddings, GLOVE_DIM)
        synsets = wn.synsets(lemma_str)

        if not synsets:
            mlp_dev_preds[inst_id] = None
            continue

        best_synset = None
        best_score  = -1e9
        
        for syn in synsets:
            sense_words = get_synset_text_words(syn)
            sense_vec   = avg_vector(sense_words, embeddings, GLOVE_DIM)
            feat        = make_feature_m4(ctx_vec, sense_vec)
            prob, _, _, _ = trained_mlp.forward(feat)
            if prob > best_score:
                best_score  = prob
                best_synset = syn
        mlp_dev_preds[inst_id] = best_synset

    mlp_dev_acc = evaluate(mlp_dev_preds, dev_key)

    print(f"  Dev  accuracy: {mlp_dev_acc:.4f}")
    print()

    # Summary table
    print("=" * 62)
    print(f"{'Method':<36} {'Dev':>10} {'Test':>10}")
    print("-" * 62)
    print(f"{'1. Most Frequent Sense (baseline)':<36} {mfs_dev_acc:>10.4f} {mfs_test_acc:>10.4f}")
    print(f"{'2. NLTK Lesk (baseline)':<36} {lesk_dev_acc:>10.4f} {lesk_test_acc:>10.4f}")
    print(f"{'3. GloVe+MLP unsupervised (neural 1)':<36} {glove_mlp_dev_acc:>10.4f} {glove_mlp_test_acc:>10.4f}")
    print(f"{'4. GloVe+MLP supervised  (neural 2)':<36} {mlp_dev_acc:>10.4f} {mlp_test_acc:>10.4f}")
    print("=" * 62)

    #  Sample output for report analysis 
    print()
    print("Sample predictions on first 5 dev instances:")
    print("-" * 70)
    
    for sid in list(dev_instances.keys())[:5]:
        inst = dev_instances[sid]
        lemma_str   = to_str(inst.lemma)
        gold_syns   = [str(sense_key_to_synset(sk)) for sk in dev_key[sid]]
        ctx_preview = ' '.join(to_str(w) for w in inst.context[:8])

        print(f"ID: {sid}  |  Lemma: {lemma_str}")
        print(f"  Context (first 8): {ctx_preview} ...")
        print(f"  Gold:               {gold_syns}")
        print(f"  MFS:                {mfs_dev_preds.get(sid)}")
        print(f"  Lesk:               {lesk_dev_preds.get(sid)}")
        print(f"  GloVe+MLP (M3):     {glove_mlp_dev_preds.get(sid)}")
        print(f"  Supervised MLP(M4): {mlp_dev_preds.get(sid)}")
        print()
