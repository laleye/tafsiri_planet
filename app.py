import os
import streamlit as st
from detection import predict, load_model


model = load_model('files/best_audio_model.pth')


# save sound file, uploaded before, in a folder
def save_file(sound_file):

    # save your sound file in the right folder by following the path
    with open(os.path.join('files/', sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())

    return sound_file.name



def language_detector():
    st.write('## Language detection')

    st.write('### Choose a sound file in .wav format')

    # upload sound
    uploaded_file = st.file_uploader(' ', type='wav')

    if uploaded_file is not None:

        # view details
        file_details = {'filename':uploaded_file.name, 'filetype':uploaded_file.type, 'filesize':uploaded_file.size}
        st.write(file_details)

        # read and play the audio file
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')

        # save_file function
        save_file(uploaded_file)

        # define the filename
        sound = uploaded_file.name

        st.write('### Classification results')

        # if you select the predict button
        if st.button('Predict'):
            # write the prediction: the prediction of the last sound sent corresponds to the first column
            preds = predict(sound, model)
            st.write(f"We detected: Fongbe({round(preds[0], 2)}), Swahili({round(preds[1], 2)}), Wolof({round(preds[2], 2)})")



if __name__ == '__main__':
    st.header("Tafsiri - Language detection")

    st.subheader("Modèle pour la détection des langues Fongbe, Shawili ou le Wolof !!!")
    st.write('___')
    select = st.sidebar.selectbox('', ['Audio file', 'Recording'], key='1')
    st.sidebar.write(select)

    if select == 'Audio file':
        language_detector()
    elif select == 'Recording':
        st.subheader("Essayer le modèle en temps réel")
        st.sidebar.title("Paramètres")
        duration = st.sidebar.slider("Durée de l'enregistrement", 0.0, 10.0, 5.0)

        if st.button("Commencer l'enregistrement"):
            with st.spinner("Recording..."):
                prediction = record_and_predict(duration=duration)
                st.write("**Prediction**: ", prediction[0])
                st.write("**Spell Check**: ", spell_check(prediction[0]))

