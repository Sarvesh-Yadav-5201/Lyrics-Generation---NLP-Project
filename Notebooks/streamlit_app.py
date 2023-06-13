# importing the libraries:
import pandas as pd
import tensorflow 
import numpy as np
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import matplotlib.pyplot as plt
from plotly import graph_objs as go

#################################################################################################################
# Define page title and favicon
st.set_page_config(page_title="AI Lyrics Generation", page_icon=":microphone:", layout='wide')
#################################################################################################################

genre_artist = {
    'Country' : ['Luke_Combs', 'Johnny_Cash', 'John_Denver', 'Dolly_Parton', 'Morgan_Wallen'],
    'Rhythm & Blues'      : ['The_Weekend','Chris_Brown','Dua_Lipa','Ed_Sheeran','Justin_Bieber'],
    'Rock'    : ['Queen', 'The_Beatles', 'Pink_Floyd', 'Maroon5', 'Cold_Play'],
    'Pop'     : ['Taylor_Swift',  'Ariana_Grande', 'Rihanna', 'Ed_Sheeran', 'Lana_Del_Rey'],
    'Rap'     : ['Drake', 'Eminem', 'Kanye_West', 'Kendrick_Lamar', 'Nicki_Minaj'],
    'Miscellaneous'    : ['Scott_Cawthon','Emily_Dickinson', 'Robert_Burns']}


# Load the pre-trained GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = TFAutoModelForCausalLM.from_pretrained("gpt2")

# FUNCTIONS 
## Getting Prediction from trained Model:

def get_predictions (prompt,model, tokenizer,
                     max_length = 250,
                     num_beams = 1,
                     temperature = 0.6,
                     no_repeat_ngram_size=3,
                     num_return_sequences=1 ):
  
    prompt = prompt.lstrip().rstrip()
    # encoding the input text
    input_ids = tokenizer([prompt])

    # Getting output:
    beam_output = model.generate(   input_ids['input_ids'],
                                    max_length = max_length,
                                    num_beams = num_beams,
                                    temperature = temperature,
                                    no_repeat_ngram_size=no_repeat_ngram_size,
                                    num_return_sequences=num_return_sequences
                                  )
    # decoding the output tokens
    output_sequences = [tokenizer.decode(seq, clean_up_tokenization_spaces=True) for seq in beam_output]

    # Return the generated lyrics
    return output_sequences[0]

def post_processing(lyrics):
    lines = lyrics.split("\n")
    processed_lyrics = []
    current_line = ""

    for line in lines:
        words = ' '.join([word for word in line.split() if "<" not in word])
        if len(words.split()) == 1:
            current_line += " " + words
        else:
            if current_line:
                processed_lyrics.append(current_line.strip())
                current_line = ""
            processed_lyrics.append(words)

    # Check if the last line is a single-word line
    if current_line:
        processed_lyrics.append(current_line.strip())
    
    # Remove empty lines from the processed lyrics
    processed_lyrics = [line for line in processed_lyrics if line]

    for i in range(0 , len(processed_lyrics)-1):
      if len(processed_lyrics[i].split(' ')) ==1:
        processed_lyrics[i] = (processed_lyrics[i]+',' + processed_lyrics[i+1])

    return '\n'.join(processed_lyrics)

# Function to display the generated song
def displaySong(song, prompt):
    # Limit the number of lines to display
    song_lines = song.split('\n')[:25]
    song = 'START!\n' + ('\n'.join(song_lines)) + '\nEND.'

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(wspace=0.3)

    # Set the background color
    fig.patch.set_facecolor('#2b2b2b')
    ax.set_facecolor('#2b2b2b')

    # Hide the axes and spines
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set the text box properties
    text_box = ax.text(0.5, 0.5, song,
                       fontsize=11, c='w',
                       ha='center', va='center', linespacing=1.5,
                       bbox=dict(facecolor='gray', alpha=0.5, boxstyle='round,pad=0.5', linewidth=3, edgecolor='r'))

    # Set the title properties
    fig.suptitle(f'Prompt: {prompt}', x=0.5, y=0.9, ha='center', va='center', fontsize=15, fontweight='bold', color='white')

    # Display the figure using st.pyplot()
    st.pyplot(plt)
##################################################################################################################

# Define app header
header = st.container()
with header:
    st.title('AI GENERATED LYRICS')
    st.write('____________________________________')

# Define sidebar
sidebar = st.container()
with sidebar:
    # Create a navigation bar:
    nav  = st.sidebar.radio("Navigation",['Generate Lyrics'])

# Define main content area
main = st.container()
with main:
    if nav == 'Generate Lyrics':
        st.header('Enter the Prompt:')
        prompt = st.text_input('Bigger prompts, better results')
        
        genre = st.selectbox('Select a Specific Genre', list(genre_artist.keys()))
        explicit_content = st.checkbox('Show Explicit Content')

        with st.container():
            if genre == 'Miscellaneous':
                # Select Specific Artist
                artist  = st.selectbox('Select Artist',genre_artist['Miscellaneous'])
            else:
                artist  = st.selectbox('Select Artist', genre_artist[genre], index=0)

        button = st.button('Gererate Lyrics')

        if button:
            # Getting the Lyrics Generation based on selected Genre , Artist and Promt:
            path = f"SAVED MODELS\Artist Models\{genre}_models\{artist}_weights.h5"
            model.load_weights(path)

            # GETTING THE PREDICTIONS:
            output  = get_predictions(prompt , model , tokenizer )
            # Post-Precessing the results.
            output = post_processing(output)
            # Displaying the Generated Lyrics:
            displaySong(output, prompt)

