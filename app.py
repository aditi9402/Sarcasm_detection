import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

# Load the custom pre-trained BERT model and tokenizer
custom_model = TFBertForSequenceClassification.from_pretrained("bert_sarcasm_model")
custom_tokenizer = BertTokenizer.from_pretrained("bert_sarcasm_tokenizer")

# Streamlit app
st.set_page_config(
    page_title="Sarcasm Detection App",
    page_icon=":smile:",
    layout="wide"
)

# Title and Subtitle
st.title("Sarcasm Detection App")
st.markdown("### Enter a sentence and click 'Predict' to check for sarcasm.")

# Input text box for user input
user_input = st.text_area("Enter a sentence:")

# Prediction button
if st.button("Predict"):
    # Check if the user has entered any text
    if user_input:
        # Tokenize the user input using the custom BERT tokenizer
        inputs = custom_tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='tf'
        )

        # Make a prediction using the custom loaded model
        prediction = custom_model.predict(
            {
               'input_ids': inputs['input_ids'],
               'attention_mask': inputs['attention_mask']
            }
        )

        # Extract the logits from the model output
        logits = prediction[0]

        # Get the predicted probability of sarcasm
        sarcasm_prob = tf.nn.softmax(logits).numpy()[0][1]

        # Display the result with styling
        st.subheader("Prediction Result:")
        st.markdown("---")
        st.write(f"**Predicted Sarcasm Probability:** {sarcasm_prob:.2%}")
        
        # Display a humorous message based on the prediction
        if sarcasm_prob > 0.75:
            st.warning("This sentence seems sarcastic! 😄")
        elif sarcasm_prob < 0.5:
            st.info("This sentence seems regular. 😐")
        else:
            st.info("The model is unsure about sarcasm in this sentence. 🤔")
        st.markdown("---")
