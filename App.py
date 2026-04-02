import streamlit as st 
import pandas as pd 
 
st.title("Healthcare Risk Dashboard") 
 
df = pd.read_csv("predictions.csv") 
 
patient_id = st.selectbox("Select Patient", df['patient_id']) 
 
row = df[df['patient_id'] == patient_id].iloc[0] 
 
st.metric("Risk Score", round(row['risk_score'], 3)) 
 
st.write("### Clinical Profile") 
st.write("Diagnosis:", row['diagnosis']) 
st.write("Claims:", row['prior_claims']) 
st.write("Drug Count:", row['avg_drug_count']) 
 
st.write("### Model Explanation") 
st.write(row.get('llm_explanation', 'Not generated')) 
[optional/bonus]LANGCHAIN EXPLANATION LAYER 
from langchain.chat_models import ChatOpenAI 
from langchain.prompts import PromptTemplate 
 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
 
prompt = PromptTemplate( 
    input_variables=["claims","drug","diagnosis","risk"], 
    template=""" 
Explain patient risk: 
 
Diagnosis: {diagnosis} 
Claims: {claims} 
Drug count: {drug} 
Risk score: {risk} 
 
Give concise clinical reasoning. 
""" 
) 
 
def generate_explanation(row): 
    return llm.predict(prompt.format( 
        claims=row['prior_claims'], 
        drug=row['avg_drug_count'], 
        diagnosis=row['diagnosis'], 
        risk=row['risk_score'] 
    ))