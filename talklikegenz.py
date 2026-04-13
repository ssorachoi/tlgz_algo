# ----------------- Setup -----------------
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk
from tkinter import ttk, messagebox
import re
import warnings

CSV_PATH = "genz_dataset_final_augmented (1).csv"

# ----------------- Color Palette -----------------
COLORS = {
    'bg_primary': '#ffffff',      # White
    'bg_secondary': '#66bb6a',    # Medium green
    'bg_card': '#f1f8f4',         # Off-white green tint
    'accent': '#4caf50',          # Green
    'accent_light': '#81c784',    # Lighter green
    'text_primary': '#ffffff',    # White text
    'text_secondary': '#ffffff',  # White text
    'success': '#4caf50',         # Success green
    'hover': '#43a047',           # Darker hover green
    'border': '#c8e6c9'           # Light green border
}

# ----------------- Load CSV -----------------
# Column mapping for different dataset schemas
SLANG_COL = "Slang"
DEF_COL = "Definition"
EX_COL = "Example Sentence"
try:
    df = pd.read_csv(CSV_PATH)
    print("CSV loaded successfully.")
    print(f"Columns in CSV: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    # Check first row to see data structure
    if len(df) > 0:
        print(f"First row example: {df.iloc[0][EX_COL] if EX_COL in df.columns else 'No Example column'}")
except FileNotFoundError:
    print("CSV not found. Creating new CSV.")
    df = pd.DataFrame(columns=[SLANG_COL, DEF_COL, EX_COL])
    df.to_csv(CSV_PATH, index=False)

# Clean and validate data (no need for column mapping - dataset is already in correct format)
df = df.drop_duplicates(subset=SLANG_COL, keep='first')
df = df.dropna(subset=[SLANG_COL, DEF_COL])
df[SLANG_COL] = df[SLANG_COL].astype(str).str.strip()
df[DEF_COL] = df[DEF_COL].astype(str).str.strip()
df[EX_COL] = df[EX_COL].astype(str).str.strip() if EX_COL in df.columns else ""
df = df[(df[SLANG_COL] != "") & (df[DEF_COL] != "")]

# ================= Manual Meaning Corrections =================
# WHY Manual Corrections?
# - Dataset contains placeholder/circular definitions (e.g., "ate" → "ate")
# - These occur in duplicated or auto-generated entries
# - Use context from highest-quality entries to infer real meanings
# - Improves user experience significantly
#
# CORRECTION STRATEGY: Map placeholder definitions to meaningful ones
MEANING_CORRECTIONS = {
    'ate': 'did really well or performed excellently',
    'bare minimum': 'the lowest level of effort required',
    'caught in 4k': 'caught on camera with undeniable proof',
    'hard launch': 'publicly announcing something officially',
    'ick': 'an immediate feeling of disgust or discomfort',
    'big yikes': 'a strong expression of disapproval',
    'head empty': 'not thinking about anything important',
}

def correct_meaning(term_key, meaning):
    """
    Apply corrections to placeholder or circular definitions.
    
    Args:
        term_key (str): The lowercase slang term
        meaning (str): The original meaning from dataset
    
    Returns:
        str: Corrected meaning or original if no correction needed
    """
    # Check if meaning is a placeholder (equals the term or very short)
    if meaning.lower() == term_key or len(meaning) < 3:
        return MEANING_CORRECTIONS.get(term_key, meaning)
    return meaning

# ================= Create Hash Table for O(1) Lookups =================
# WHY Hash Table Instead of Linear Search?
# - Direct access: O(1) average case vs O(n) for linear search
# - Scales efficiently with dictionary growth
# - Perfect for repeated lookups in interactive applications
def build_hash_table():
    """
    Build hash table for O(1) dictionary lookups by extracting best examples.
    
    Quality Filtering Strategy:
    - For duplicate slang terms, prefer entries with:
      (1) Longer, more descriptive examples (more informative)
      (2) "real" or "viral" trend_status over "generated"
      (3) Higher frequency_score for commonly used terms
    
    Time Complexity: O(n log n) with sorting for quality ranking
    Space Complexity: O(n) for hash table storage
    
    Returns:
        dict: Hash table with lowercase slang terms as keys,
              (meaning, example) tuples as values
    """
    slang_dict = {}
    
    for _, row in df.iterrows():
        key = str(row[SLANG_COL]).strip().lower()
        meaning = str(row[DEF_COL]).strip()
        example = str(row[EX_COL]).strip() if EX_COL in df.columns else "N/A"
        
        # Get quality scores for better example selection
        trend_status = str(row['trend_status']).strip().lower() if 'trend_status' in df.columns else 'unknown'
        freq_score = float(row['frequency_score']) if 'frequency_score' in df.columns and pd.notna(row['frequency_score']) else 0.0
        
        # Quality heuristic: length + trend + frequency
        quality_score = len(example) + (10 if trend_status in ['real', 'viral'] else 0) + (freq_score / 10)
        
        if key not in slang_dict:
            # First entry for this term
            slang_dict[key] = {
                'meaning': meaning,
                'example': example,
                'quality': quality_score
            }
        else:
            # Replace if new entry has higher quality
            if quality_score > slang_dict[key]['quality']:
                slang_dict[key] = {
                    'meaning': meaning,
                    'example': example,
                    'quality': quality_score
                }
    
    # Flatten to (meaning, example) tuples and apply corrections
    result = {}
    for k, v in slang_dict.items():
        meaning = correct_meaning(k, v['meaning'])
        result[k] = (meaning, v['example'])
    return result

slang_dict = build_hash_table()
model = None
vectorizer = None

# ================= ALGORITHM EXPLANATIONS =================
# HASH TABLE (Dictionary Lookup): O(1) Time Complexity
# ===========================================================
# WHY Hash Table?
# - Constant-time O(1) average-case lookups vs O(n) linear search
# - Essential for scalable dictionary applications with dynamic datasets
# - Industry standard for word/key lookup in NLP tools
# - Supports efficient updates when adding new slang terms
#
# IMPLEMENTATION: Python dict provides native hash table with O(1) lookup
# - Key: lowercase slang term (normalized for case-insensitive matching)
# - Value: tuple of (meaning, example_sentence)
#
# KNUTH-MORRIS-PRATT (KMP) Algorithm: O(n + m) Time Complexity
# ===========================================================
# WHY KMP for Advanced Substring Matching?
# - Naive string matching: O(n*m) with redundant comparisons
# - KMP algorithm: O(n + m) with failure function preprocessing
# - Prevents character re-comparisons by using pattern structure
# - Optimal for finding partial slang matches or pattern detection
# - Better than naive approach for long text or multiple searches

def build_kmp_failure_function(pattern):
    """
    Build KMP failure function (prefix table) for pattern matching.
    Time: O(m) where m is pattern length
    Space: O(m)
    
    Args:
        pattern (str): The pattern to build failure function for
    
    Returns:
        list: Failure function array for KMP algorithm
    """
    m = len(pattern)
    failure = [0] * m
    j = 0
    
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        failure[i] = j
    
    return failure

def kmp_search(text, pattern):
    """
    Knuth-Morris-Pratt string matching algorithm.
    Time: O(n + m) where n = text length, m = pattern length
    Space: O(m) for failure function
    
    Args:
        text (str): The text to search in
        pattern (str): The pattern to find
    
    Returns:
        list: All starting indices where pattern is found in text
    """
    n = len(text)
    m = len(pattern)
    
    if m == 0 or m > n:
        return []
    
    failure = build_kmp_failure_function(pattern)
    matches = []
    j = 0
    
    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = failure[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - m + 1)
            j = failure[j - 1]
    
    return matches

def normalize_text(text):
    """
    Normalize user input for reliable exact and substring matching.
    """
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def add_unknown_slang(word):
    """
    Interactive flow to add unknown slang to CSV, then refresh model and hash table.
    """
    is_slang = messagebox.askyesno("Unknown Slang", f"'{word}' is unknown. Is this a slang?")
    if not is_slang:
        return f"⏭️ '{word}' skipped."

    meaning = simple_input(f"Enter meaning of '{word}':")
    if not meaning or not meaning.strip():
        return f"⏭️ '{word}' skipped."

    example = simple_input(f"Enter example sentence for '{word}' (or leave blank):")
    if not example:
        example = "N/A"

    new_row = {
        SLANG_COL: word,
        DEF_COL: meaning,
        EX_COL: example,
    }

    df_temp = pd.read_csv(CSV_PATH)
    df_temp = pd.concat([df_temp, pd.DataFrame([new_row])], ignore_index=True)
    df_temp.to_csv(CSV_PATH, index=False)

    retrain_model()
    return f"✅ '{word}' added to dictionary!"

def find_kmp_candidates(query, max_results=5):
    """
    Use KMP to find close dictionary terms when exact lookup fails.

    Matching strategy:
    - query is substring of term (term contains query)
    - term is substring of query (query contains term)
    """
    q = normalize_text(query)
    if not q:
        return []

    scored = []
    for term in slang_dict.keys():
        contains_query = len(kmp_search(term, q)) > 0
        query_contains_term = len(kmp_search(q, term)) > 0
        if contains_query or query_contains_term:
            # Smaller length gap usually means closer match.
            length_gap = abs(len(term) - len(q))
            scored.append((length_gap, term))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [term for _, term in scored[:max_results]]

def predict_with_naive_bayes(query, confidence_threshold=0.35):
    """
    Fallback prediction for unresolved queries using Multinomial Naive Bayes.
    Returns (label, confidence) when confidence passes threshold, else None.
    """
    if model is None or vectorizer is None:
        return None

    q = normalize_text(query)
    if not q:
        return None

    try:
        q_vec = vectorizer.transform([q])
        pred_label = model.predict(q_vec)[0]
        proba = model.predict_proba(q_vec)[0]
        confidence = float(proba.max())
        if confidence >= confidence_threshold:
            return pred_label, confidence
    except Exception:
        return None

    return None

def translate_query(user_text):
    """
    Query pipeline aligned with project goals:
    1) Exact hash-table lookup (O(1) average)
    2) Phrase scan via hash lookups
    3) KMP candidate suggestions (O(n + m) per comparison)
    4) Naive Bayes probabilistic fallback for unresolved queries
    """
    raw = user_text.strip()
    normalized = normalize_text(raw)
    if not normalized:
        return "Please enter a slang word or sentence."

    # Stage 1: direct dictionary lookup for single terms.
    if " " not in normalized and normalized in slang_dict:
        meaning, example = slang_dict[normalized]
        if isinstance(example, str):
            example = example.strip().strip('"').strip("'")
        return f"{raw} → {meaning}\n📝 Example: {example}"

    # Stage 2: sentence/phrase matching using O(1) hash lookups.
    phrase_result = translate_sentence(raw)
    if phrase_result != "❌ No slang found in the sentence.":
        return phrase_result

    # Stage 3: KMP-based candidate suggestions.
    kmp_candidates = find_kmp_candidates(normalized, max_results=5)
    if kmp_candidates:
        lines = ["🔎 No exact match, but here are close slang terms (KMP):"]
        for term in kmp_candidates:
            meaning, _ = slang_dict[term]
            lines.append(f"• {term} → {meaning}")
        return "\n".join(lines)

    # Stage 4: probabilistic ML fallback.
    nb_guess = predict_with_naive_bayes(normalized)
    if nb_guess:
        pred_label, confidence = nb_guess
        return (
            "🤖 No exact dictionary match found.\n"
            f"Naive Bayes fallback guess: {pred_label}\n"
            f"Confidence: {confidence:.1%}"
        )

    # Optional interactive extension only for unknown single-term input.
    if " " not in normalized:
        return add_unknown_slang(raw)

    return "❌ No slang found, and no confident fallback suggestion was available."

def retrain_model():
    """
    Retrain Naive Bayes classifier and rebuild hash table.
    Reloads CSV to ensure new entries are included.
    
    WHY Naive Bayes for Classification?
    - Good baseline for multi-class text classification
    - Fast training and prediction O(m) where m = vocabulary size
    - Works well with small to medium-sized datasets
    - Probabilistic framework suitable for slang categorization
    """
    global model, vectorizer, X_vec, slang_dict, df
    
    # Reload CSV to get newly added entries
    df = pd.read_csv(CSV_PATH)
    
    # Ensure columns exist and clean data
    if SLANG_COL in df.columns and DEF_COL in df.columns:
        df = df.drop_duplicates(subset=SLANG_COL, keep='first')
        df = df.dropna(subset=[SLANG_COL, DEF_COL])
        df[SLANG_COL] = df[SLANG_COL].astype(str).str.strip()
        df[DEF_COL] = df[DEF_COL].astype(str).str.strip()
        df = df[(df[SLANG_COL] != "") & (df[DEF_COL] != "")]
    
    # Train Naive Bayes
    X = df[SLANG_COL]
    y = df[DEF_COL]
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(X)
    model = MultinomialNB()
    # This dataset maps many unique definitions to slang entries, which can
    # trigger sklearn's high-class-cardinality warning. The model is still
    # used only as a low-confidence fallback, so we silence that expected noise.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50% of the number of samples.*",
            category=UserWarning,
        )
        model.fit(X_vec, y)
    
    # Rebuild hash table with new data
    slang_dict = build_hash_table()

def translate_word(word):
    """
    Translate a single slang word using O(1) hash table lookup.
    
    Algorithm: Direct hash table access
    Time Complexity: O(1) average case for lookup
    Space Complexity: O(1)
    
    Args:
        word (str): The slang word to translate
    
    Returns:
        str: Translation result with meaning and example
    """
    w_clean = word.strip().lower()
    
    # O(1) hash table lookup instead of O(n) dataframe filtering
    if w_clean in slang_dict:
        meaning, example = slang_dict[w_clean]
        # Clean up extra quotes from CSV
        if isinstance(example, str):
            example = example.strip().strip('"').strip("'")
        return f"{word} → {meaning}\n📝 Example: {example}"

    return add_unknown_slang(word)

def translate_sentence(sentence):
    """
    Translate all slang words and multi-word phrases in a sentence.
    
    Algorithm: Greedy multi-word phrase matching with O(1) hash table lookups
    - Prioritizes longer phrases (e.g., "down bad" before "down")
    - Greedy approach: matches longest available phrase first
    Time Complexity: O(k*m) where k = words, m = max phrase length
    Space Complexity: O(k) for results storage
    
    WHY This Approach?
    - Handles multi-word slang terms (e.g., "down bad", "so down bad af")
    - Greedy prioritization prevents incorrect single-word matches
    - Efficient without complex NLP pipeline
    
    Args:
        sentence (str): The sentence containing slang to translate
    
    Returns:
        str: Formatted translation results for all found slang terms
    """
    words = sentence.split()
    results = []
    i = 0
    max_phrase_len = 5  # Max words in a phrase (optimization)

    while i < len(words):
        found = False
        
        # Try to match longest phrases first (greedy approach)
        for phrase_len in range(min(max_phrase_len, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i+phrase_len]).lower().strip()
            
            # O(1) hash table lookup
            if phrase in slang_dict:
                meaning, example = slang_dict[phrase]
                
                if isinstance(example, str):
                    example = example.strip().strip('"').strip("'")
                
                # Display original casing for the phrase
                original_phrase = " ".join(words[i:i+phrase_len])
                results.append(f"💬 {original_phrase} → {meaning}\n   📝 Example: {example}\n")
                
                i += phrase_len
                found = True
                break
        
        if not found:
            i += 1  # Move to next word if no phrase found

    if results:
        return "\n".join(results)
    else:
        return "❌ No slang found in the sentence."

def simple_input(prompt):
    input_win = tk.Toplevel(root)
    input_win.title("Input Required")
    input_win.geometry("450x180")
    input_win.configure(bg=COLORS['bg_secondary'])
    input_win.resizable(False, False)
    
    # Center the window
    input_win.transient(root)
    input_win.grab_set()
    
    frame = tk.Frame(input_win, bg=COLORS['bg_secondary'])
    frame.pack(expand=True, fill='both', padx=20, pady=20)
    
    label = tk.Label(frame, text=prompt, font=("Segoe UI", 11), 
                    bg=COLORS['bg_secondary'], fg=COLORS['text_primary'], wraplength=400)
    label.pack(pady=(0, 15))
    
    entry = tk.Entry(frame, width=50, font=("Segoe UI", 10), 
                    bg='white', fg='#2e7d32',
                    insertbackground='#2e7d32', relief='solid',
                    highlightthickness=1, highlightbackground=COLORS['border'],
                    highlightcolor=COLORS['accent'], bd=1)
    entry.pack(pady=(0, 15), ipady=8)
    entry.focus()

    result = {"value": None}

    def submit():
        result["value"] = entry.get()
        input_win.destroy()

    submit_btn = tk.Button(frame, text="Submit", command=submit,
                          bg=COLORS['accent'], fg=COLORS['text_primary'],
                          font=("Segoe UI", 10, "bold"), relief='flat',
                          cursor='hand2', padx=30, pady=8,
                          activebackground=COLORS['hover'],
                          activeforeground=COLORS['text_primary'])
    submit_btn.pack()
    
    entry.bind('<Return>', lambda e: submit())
    
    root.wait_window(input_win)
    return result["value"]

# ----------------- GUI -----------------
root = tk.Tk()
root.title("TalkLikeGenZ Translator")
root.geometry("700x650")
root.resizable(False, False)
root.configure(bg=COLORS['bg_primary'])

# Header Frame
header_frame = tk.Frame(root, bg=COLORS['bg_secondary'], height=100)
header_frame.pack(fill='x', pady=(0, 20))
header_frame.pack_propagate(False)

# Create canvas for rounded corners
canvas = tk.Canvas(header_frame, bg=COLORS['bg_primary'], height=100, highlightthickness=0)
canvas.pack(fill='both', expand=True)

# Draw rounded rectangle
def round_rectangle(x1, y1, x2, y2, radius=25):
    points = [
        x1+radius, y1,
        x1+radius, y1,
        x2-radius, y1,
        x2-radius, y1,
        x2, y1,
        x2, y1+radius,
        x2, y1+radius,
        x2, y2-radius,
        x2, y2-radius,
        x2, y2,
        x2-radius, y2,
        x2-radius, y2,
        x1+radius, y2,
        x1+radius, y2,
        x1, y2,
        x1, y2-radius,
        x1, y2-radius,
        x1, y1+radius,
        x1, y1+radius,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, fill=COLORS['bg_secondary'], outline=COLORS['bg_secondary'])

canvas.create_rectangle(0, 0, 700, 100, fill=COLORS['bg_secondary'], outline='')

title = tk.Label(canvas, text="TalkLikeGenZ", 
                font=("Segoe UI", 28, "bold"),
                bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
canvas.create_window(350, 40, window=title)

subtitle = tk.Label(canvas, text="Decode Gen Z Slang Instantly", 
                   font=("Segoe UI", 12),
                   bg=COLORS['bg_secondary'], fg=COLORS['text_secondary'])
canvas.create_window(350, 75, window=subtitle)

# Main Container
main_frame = tk.Frame(root, bg=COLORS['bg_primary'])
main_frame.pack(expand=True, fill='both', padx=30, pady=(0, 30))

# Input Section
input_container = tk.Frame(main_frame, bg=COLORS['bg_card'], relief='flat', highlightthickness=0)
input_container.pack(fill='x', pady=(0, 20))

# Add rounded corners effect with padding
input_inner = tk.Frame(input_container, bg=COLORS['bg_card'])
input_inner.pack(padx=2, pady=2, fill='both', expand=True)

input_label = tk.Label(input_inner, text="Enter slang word or sentence:", 
                      font=("Segoe UI", 11, "bold"),
                      bg=COLORS['bg_card'], fg='#2e7d32')
input_label.pack(anchor='w', padx=20, pady=(15, 8))

entry = tk.Entry(input_inner, width=50, font=("Segoe UI", 12),
                bg='white', fg='#2e7d32',
                insertbackground='#2e7d32', relief='solid',
                highlightthickness=1, highlightbackground=COLORS['border'],
                highlightcolor=COLORS['accent'], bd=1)
entry.pack(fill='x', padx=20, pady=(0, 15), ipady=12)

# Translate Button
def on_enter_hover(e):
    pass

def on_leave_hover(e):
    pass

# Translate Button with rounded corners
button_canvas = tk.Canvas(input_inner, width=200, height=50, bg=COLORS['bg_card'], highlightthickness=0)
button_canvas.pack(pady=(0, 20))

def create_rounded_button(canvas, x, y, width, height, radius, color):
    points = [
        x+radius, y,
        x+width-radius, y,
        x+width, y,
        x+width, y+radius,
        x+width, y+height-radius,
        x+width, y+height,
        x+width-radius, y+height,
        x+radius, y+height,
        x, y+height,
        x, y+height-radius,
        x, y+radius,
        x, y
    ]
    return canvas.create_polygon(points, smooth=True, fill=color, outline='')

btn_rect = create_rounded_button(button_canvas, 10, 5, 180, 40, 20, COLORS['accent'])
btn_text = button_canvas.create_text(100, 25, text="🔍 Translate", 
                                      font=("Segoe UI", 12, "bold"), 
                                      fill=COLORS['text_primary'])

def on_button_click(event):
    on_translate()

def on_button_enter_new(event):
    button_canvas.itemconfig(btn_rect, fill=COLORS['hover'])
    button_canvas.config(cursor='hand2')

def on_button_leave_new(event):
    button_canvas.itemconfig(btn_rect, fill=COLORS['accent'])
    button_canvas.config(cursor='')

def add_slang_manually():
    """
    Function to manually add slang to the CSV.
    Uses O(1) hash table lookup to check for duplicates.
    Works with simplified dataset format: Slang, Definition, Example Sentence
    """
    global df
    
    slang_word = simple_input("Enter the slang word:")
    if not slang_word or not slang_word.strip():
        return
    
    # O(1) hash table lookup instead of O(n) dataframe filtering
    w_clean = slang_word.strip().lower()
    if w_clean in slang_dict:
        messagebox.showinfo("Already Exists", f"'{slang_word}' is already in the dictionary!")
        return
    
    meaning = simple_input(f"Enter the meaning of '{slang_word}':")
    if not meaning or not meaning.strip():
        messagebox.showwarning("Input Error", "Meaning cannot be empty!")
        return
    
    example = simple_input(f"Enter an example sentence for '{slang_word}' (or leave blank):")
    if not example or not example.strip():
        example = "N/A"
    
    # Create new row with simplified format
    new_row = {
        SLANG_COL: slang_word,
        DEF_COL: meaning,
        EX_COL: example,
    }
    
    # Add to dataframe
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    retrain_model()
    
    messagebox.showinfo("Success", f"✅ '{slang_word}' has been added to the dictionary!")
    
    # Update output text
    output_text.config(state='normal')
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"✅ Successfully added:\n\n💬 {slang_word} → {meaning}\n📝 Example: {example}")
    output_text.config(state='disabled')

button_canvas.bind('<Button-1>', on_button_click)
button_canvas.bind('<Enter>', on_button_enter_new)
button_canvas.bind('<Leave>', on_button_leave_new)

# Add Slang Button with rounded corners
add_button_canvas = tk.Canvas(input_inner, width=200, height=50, bg=COLORS['bg_card'], highlightthickness=0)
add_button_canvas.pack(pady=(0, 20))

add_btn_rect = create_rounded_button(add_button_canvas, 10, 5, 180, 40, 20, COLORS['accent'])
add_btn_text = add_button_canvas.create_text(100, 25, text="➕ Add Slang", 
                                              font=("Segoe UI", 12, "bold"), 
                                              fill=COLORS['text_primary'])

def on_add_button_click(event):
    add_slang_manually()

def on_add_button_enter(event):
    add_button_canvas.itemconfig(add_btn_rect, fill=COLORS['hover'])
    add_button_canvas.config(cursor='hand2')

def on_add_button_leave(event):
    add_button_canvas.itemconfig(add_btn_rect, fill=COLORS['accent'])
    add_button_canvas.config(cursor='')

add_button_canvas.bind('<Button-1>', on_add_button_click)
add_button_canvas.bind('<Enter>', on_add_button_enter)
add_button_canvas.bind('<Leave>', on_add_button_leave)

# Output Section
output_label = tk.Label(main_frame, text="Translation:", 
                       font=("Segoe UI", 11, "bold"),
                       bg=COLORS['bg_primary'], fg='#2e7d32')
output_label.pack(anchor='w', pady=(0, 8))

output_frame = tk.Frame(main_frame, bg=COLORS['bg_card'], relief='flat')
output_frame.pack(fill='both', expand=True)

output_text = tk.Text(output_frame, height=15, width=70, font=("Segoe UI", 11),
                     bg='white', fg='#2e7d32',
                     relief='solid', borderwidth=1, padx=15, pady=15,
                     wrap='word', insertbackground='#2e7d32',
                     highlightthickness=1, highlightbackground=COLORS['border'])
output_text.pack(side='left', fill='both', expand=True, padx=(15, 0), pady=15)

scrollbar = tk.Scrollbar(output_frame, command=output_text.yview,
                        bg=COLORS['bg_card'], troughcolor=COLORS['bg_primary'],
                        activebackground=COLORS['accent'])
scrollbar.pack(side='right', fill='y', padx=(0, 5), pady=15)
output_text.config(yscrollcommand=scrollbar.set)

# Initial message
output_text.insert(tk.END, "👋 Welcome! Enter a Gen Z slang word or sentence above to translate.\n\n")
output_text.insert(tk.END, "💡 Tip: If a word isn't in the dictionary, you can add it!")
output_text.config(state='disabled')

def on_translate():
    text = entry.get().strip()
    if text:
        output_text.config(state='normal')
        output_text.delete(1.0, tk.END)
        result = translate_query(text)
        print(f"Final result to display:\n{result}")  # Debug output
        output_text.insert(tk.END, result)
        output_text.config(state='disabled')
    else:
        messagebox.showwarning("Input Error", "Please enter a slang word or sentence.")

# Bind Enter key to translate
entry.bind('<Return>', lambda e: on_translate())

# Train ML model once at startup so fallback is available before first query.
retrain_model()

# Footer
footer = tk.Label(root, text="Made with 💚 for Gen Z", 
                 font=("Segoe UI", 9),
                 bg=COLORS['bg_primary'], fg='#2e7d32')
footer.pack(side='bottom', pady=(0, 10))

root.mainloop()