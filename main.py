import streamlit as st
import pandas as pd
import json
import os
from argparse import ArgumentParser

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import wandb_tracing_enabled
from langchain.chat_models import ChatOpenAI


st.set_page_config(page_title='Data Transformation', layout='wide')


source_file = st.file_uploader("Upload Template File")
template_uploaded = False
if source_file is not None:
    try:
        df_template = pd.read_csv(source_file)
        st.write("Template HEAD")
        st.write(df_template.head())
        df_template.to_csv("df_template.csv", index=False)
        template_uploaded = True
    except:
        st.error("Template file is required in csv format")

else:
    st.error('Template file is required')

source_file = st.file_uploader("Upload Raw File")
source_file_uploaded = False
if source_file is not None:
    try:
        df_source = pd.read_csv(source_file)
        st.write("Source HEAD")
        st.write(df_source.head())
        df_source.to_csv("df_source.csv", index=False)
        source_file_uploaded = True
    except:
        st.error("Source file is required in csv format")

else:
    st.error('Raw file is required')

open_ai_key = st.sidebar.text_input("OpenAI Key", type="password")


st.session_state.mapping_table_generated = False
st.session_state.transformation_code_generated = False
st.session_state.output_df_generated = False

if open_ai_key and template_uploaded and source_file_uploaded:
    if st.button('Show Mappings'):
        st.write("GENERATING MAPPING TABLE")

        with open("column_mapping_template.txt") as f:
            column_mapping_template = f.read()
        prompt = PromptTemplate(
            template=column_mapping_template, input_variables=[
                "df_source_rows", "df_template_rows"]
        )

        llm = ChatOpenAI(openai_api_key=open_ai_key, model="gpt-3.5-turbo")
        llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

        df_source_rows = df_source.iloc[:2].to_json()
        df_template_rows = df_template.iloc[:2].to_json()
        response = llm_chain.run(
            {"df_source_rows": df_source_rows, "df_template_rows": df_template_rows}
        )
        data = json.loads(response)
        target_df = pd.DataFrame(data)
        target_df["check_mapping"] = [True] * len(target_df)

        target_df.to_csv("mapping_table.csv", index=False)

        st.session_state.mapping_table_generated = True

    if st.session_state.mapping_table_generated:
        target_df = pd.read_csv("mapping_table.csv")

        #part of the code which can be used to take user's input on the mapping table
        # edited_df = st.data_editor(
        #     target_df,
        #     column_config={
        #         "check_mapping": st.column_config.CheckboxColumn(
        #             "check_mapping?",
        #             default=False,
        #         )
        #     },
        #     key="target_df",
        #     disabled=["st.selectbox"],
        #     hide_index=True,
        # )
        st.dataframe(target_df)
        st.session_state.mapping_table_generated = True

    if st.button('Generate Transformations'):
        edited_df = pd.read_csv("mapping_table.csv")
        df_filtered = edited_df[edited_df['check_mapping'] == True]
        mapping_table = df_filtered.drop(
            ['Basis for the decision', 'check_mapping'], axis=1)

        st.write("Tranforming Data")

        with open("data_transformation_template.txt") as f:
            data_transformation_template = f.read()

        prompt = PromptTemplate(
            template=data_transformation_template, input_variables=[
                "df_source_rows", "df_template_rows", "mapping_table"]
        )

        llm = ChatOpenAI(openai_api_key=open_ai_key, model="gpt-3.5-turbo")
        llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

        df_source_rows = df_source.iloc[:2].to_json()
        df_template_rows = df_template.iloc[:2].to_json()
        response = llm_chain.run(
            {"df_source_rows": df_source_rows,
                "df_template_rows": df_template_rows, "mapping_table": mapping_table}
        )
        st.code(response)

        f = open("generated_code.txt", "w")
        f.write(response)
        f.close()

        st.session_state.transformation_code_generated = True

    if st.session_state.transformation_code_generated:
        st.write("DISPLAYING CODE")
        with open("generated_code.txt") as f:
            code = f.read()

    if st.button('run code'):
        with open("generated_code.txt") as f:
            code = f.read()
        exec(code)

        st.session_state.output_df_generated = True

    if st.session_state.output_df_generated:
        df = pd.read_csv("output.csv")
        st.dataframe(df)

else:
    st.sidebar.error('OpenAI Api key is required')
