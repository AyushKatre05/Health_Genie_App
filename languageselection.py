import streamlit as st

# Function to display English content
def english_content():
    st.title("English Language Selected")
    st.write("Welcome to the English version of the application!")
    # Add more English content here

# Function to display Hindi content
def hindi_content():
    st.title("हिंदी भाषा चयन किया गया")
    st.write("एप्लिकेशन के हिंदी संस्करण में आपका स्वागत है!")
    # Add more Hindi content here

# Main function
def main():
    st.title("Language Selection / भाषा चयन")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("English"):
            english_content()
            st.markdown("<a href='https://diseasepredictor-health-genie.streamlit.app/'>Go to English version</a>", unsafe_allow_html=True)

    with col2:
        if st.button("Hindi"):
            hindi_content()

if __name__ == "__main__":
    main()
