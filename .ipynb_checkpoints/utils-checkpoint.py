import numpy as np
import matplotlib.pyplot as plt

def plot_tsne(words, embeds_2d):
    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words):
        plt.scatter(embeds_2d[i, 0], embeds_2d[i, 1], marker='o')
        plt.text(embeds_2d[i, 0] + 0.1, embeds_2d[i, 1], word, fontsize=12)
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("t-SNE visualization of word embeddings")
    plt.grid(True)
    plt.show()
    
    

def plot_debiasing_tsne(embeds_2d, words, gender_pairs, neutral_words, other_words, title):
    plt.figure(figsize=(10, 8))

    # colour coding
    for i, word in enumerate(words):
        if word in [w for pair in gender_pairs for w in pair]:
            color = 'red'       # gendered words - red
        elif word in neutral_words:
            color = 'blue'      # neutral words - blue
        else:
            color = 'green'     # others - green

        plt.scatter(embeds_2d[i, 0], embeds_2d[i, 1], c=color)
        plt.text(embeds_2d[i, 0]+0.3, embeds_2d[i, 1]+0.3, word, fontsize=9)

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(title)
    plt.grid(True)
    file = ''.join(title.lower().split())
    plt.savefig('images/'+file+'.png', bbox_inches='tight')
    plt.show()

def read_glove_vecs(glove_file):
    words = set()
    word_to_vec_map = {}
        
    with open(glove_file, 'r') as f:
        for line in f:
            a_line = line.strip().split()
            curr_word = a_line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(a_line[1:], dtype=np.float64)

    return words, word_to_vec_map

def word_analogy(word_a, word_b, word_c, word_to_vec_map, vocab):
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    e_a = word_to_vec_map[word_a]
    e_b = word_to_vec_map[word_b]
    e_c = word_to_vec_map[word_c]

    target_vector = e_b - e_a + e_c

    max_cosine_similarity = -np.inf
    best_word = None

    for w in vocab:
        if w in [word_a, word_b, word_c]:
            continue

        e_w = word_to_vec_map[w]
        similarity = cosine_similarity(target_vector, e_w)

        if similarity > max_cosine_similarity:
            max_cosine_similarity = similarity
            best_word = w

    return best_word

def cosine_similarity(u, v):
    uv_dot = np.dot(u,v)
    u_norm = np.sqrt(np.sum(np.square(u)))
    v_norm = np.sqrt(np.sum(np.square(v)))

    # avoid division by 0
    if np.isclose(u_norm * v_norm, 0, atol=1e-32):
        return 0
    
    return uv_dot / (u_norm * v_norm)

def neutralize(word, g, word_to_vec_map):
    e = word_to_vec_map[word]
    
    # compute projection of e onto the bias direction g
    bias_component = np.dot(e, g) / np.linalg.norm(g)**2 * g
    
    # subtract bias component to get debiased embedding
    e_debiased = e - bias_component
    
    return e_debiased

def equalize(pair, bias_axis, word_to_vec_map):
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    # mean vector
    mu = (e_w1 + e_w2) / 2

    # projection of mean vector onto bias axis
    mu_B = (np.dot(mu, bias_axis) / np.linalg.norm(bias_axis)**2) * bias_axis
    # orthogonal component of mean vector
    mu_orth = mu - mu_B

    # projection of each word vector onto bias axis
    e_w1B = (np.dot(e_w1, bias_axis) / np.linalg.norm(bias_axis)**2) * bias_axis
    e_w2B = (np.dot(e_w2, bias_axis) / np.linalg.norm(bias_axis)**2) * bias_axis

    # compute the magnitude to scale corrected projections
    correction_factor = np.sqrt(1 - np.linalg.norm(mu_orth)**2)

    # normalize vectors for corrected bias components
    corrected_e_w1B = correction_factor * (e_w1B - mu_B) / np.linalg.norm(e_w1B - mu_B)
    corrected_e_w2B = correction_factor * (e_w2B - mu_B) / np.linalg.norm(e_w2B - mu_B)

    # final equalized embeddings
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    return e1, e2