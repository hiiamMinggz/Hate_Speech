import numpy as np
import gradio as gr
from sentiment_bert import bert_sentiment_classification
from sentiment_bayes import bayes_sentiment_classification
from sentiment_gru import gru_sentiment_classification


css_string = "#title {text-align: center;}"
element_id = "title"
with gr.Blocks(css=css_string) as demo:
    gr.Markdown(elem_id = element_id, value="# Chương trình phân tích cảm xúc")
    with gr.Tab("Mô hình Bayes"):
        text_input_bayes = gr.Textbox()
        text_output_bayes = gr.Label()
        text_button_bayes = gr.Button("Submit")

    with gr.Tab("Mô hình GRU_MLP"):
        text_input_GRU = gr.Textbox()
        text_output_GRU = gr.Label()
        text_button_GRU = gr.Button("Submit")
    
    with gr.Tab("Mô hình Bert_MLP"):
        text_input_Bert = gr.Textbox()
        text_output_Bert = gr.Label()
        text_button_Bert = gr.Button("Submit")

    text_button_bayes.click(bayes_sentiment_classification, inputs=text_input_bayes, outputs=text_output_bayes)
    text_button_Bert.click(bert_sentiment_classification, inputs=text_input_Bert, outputs=text_output_Bert)
    text_button_GRU.click(gru_sentiment_classification, inputs=text_input_GRU, outputs=text_output_GRU)

if __name__ == "__main__":
    demo.launch()




