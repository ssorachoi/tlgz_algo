# ----------------- Setup -----------------
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk
from tkinter import ttk, messagebox

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

# Normalize columns for the augmented dataset schema
column_map = {
    "slang_term": SLANG_COL,
    "meaning": DEF_COL,
    "slang_sentence": EX_COL,
}
df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

df = df.drop_duplicates(subset=SLANG_COL, keep='first')
df = df.dropna(subset=[SLANG_COL, DEF_COL])
df[SLANG_COL] = df[SLANG_COL].astype(str).str.strip()
df[DEF_COL] = df[DEF_COL].astype(str).str.strip()
df = df[(df[SLANG_COL] != "") & (df[DEF_COL] != "")]

# ----------------- Train Naive Bayes -----------------
X = df[SLANG_COL]
y = df[DEF_COL]

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# ----------------- Functions -----------------
def retrain_model():
    global model, vectorizer, X_vec
    X = df[SLANG_COL]
    y = df[DEF_COL]
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vec, y)

def translate_word(word):
    w_clean = word.strip().lower()
    matched = df[df[SLANG_COL].str.lower() == w_clean]
    
    if not matched.empty:
        meaning = matched[DEF_COL].values[0]
        example = matched[EX_COL].values[0] if EX_COL in df.columns else "N/A"
        # Clean up extra quotes from CSV
        if isinstance(example, str):
            example = example.strip().strip('"').strip("'")
        return f"{word} → {meaning}\n📝 Example: {example}"
    else:
        is_slang = messagebox.askyesno("Unknown Slang", f"'{word}' is unknown. Is this a slang?")
        if is_slang:
            meaning = simple_input(f"Enter meaning of '{word}':")
            example = simple_input(f"Enter example sentence for '{word}' (or leave blank):")
            if not example:
                example = "N/A"
            df.loc[len(df)] = [word, meaning, example]
            df.to_csv(CSV_PATH, index=False)
            retrain_model()
            return f"✅ '{word}' added to dictionary!"
        else:
            return f"⏭️ '{word}' skipped."

def translate_sentence(sentence):
    words = sentence.split()
    results = []

    for word in words:
        w_clean = word.strip().lower()
        matched = df[df[SLANG_COL].str.lower() == w_clean]

        if not matched.empty:
            meaning = matched[DEF_COL].values[0]
            example = "N/A"
            
            # Debug: Print what we're getting
            print(f"Word: {word}")
            print(f"Columns available: {matched.columns.tolist()}")
            
            if EX_COL in matched.columns:
                example_value = matched[EX_COL].values[0]
                print(f"Raw example value: {repr(example_value)}")
                print(f"Is NaN: {pd.isna(example_value)}")
                
                if pd.notna(example_value):  # Check if not NaN
                    example = str(example_value).strip().strip('"').strip("'")
                    print(f"Cleaned example: {example}")
            
            results.append(f"💬 {word} → {meaning}\n   📝 Example: {example}\n")

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
    """Function to manually add slang to the CSV"""
    global df
    
    slang_word = simple_input("Enter the slang word:")
    if not slang_word or not slang_word.strip():
        return
    
    # Check if slang already exists
    w_clean = slang_word.strip().lower()
    matched = df[df[SLANG_COL].str.lower() == w_clean]
    
    if not matched.empty:
        messagebox.showinfo("Already Exists", f"'{slang_word}' is already in the dictionary!")
        return
    
    meaning = simple_input(f"Enter the meaning of '{slang_word}':")
    if not meaning or not meaning.strip():
        messagebox.showwarning("Input Error", "Meaning cannot be empty!")
        return
    
    example = simple_input(f"Enter an example sentence for '{slang_word}' (or leave blank):")
    if not example or not example.strip():
        example = "N/A"
    
    # Add to dataframe
    df.loc[len(df)] = [slang_word, meaning, example]
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
        result = translate_sentence(text)
        print(f"Final result to display:\n{result}")  # Debug output
        output_text.insert(tk.END, result)
        output_text.config(state='disabled')
    else:
        messagebox.showwarning("Input Error", "Please enter a slang word or sentence.")

# Bind Enter key to translate
entry.bind('<Return>', lambda e: on_translate())

# Footer
footer = tk.Label(root, text="Made with 💚 for Gen Z", 
                 font=("Segoe UI", 9),
                 bg=COLORS['bg_primary'], fg='#2e7d32')
footer.pack(side='bottom', pady=(0, 10))

root.mainloop()