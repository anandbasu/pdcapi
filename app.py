import streamlit as st
#from helper import get_openai_api_key
# the OEPNAI_API_KEY is set in the environment
#OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
from llama_index.core.tools import FunctionTool
import pandas as pd
#from langchain_core.tools import tool
import requests
import json
#from langchain_core.messages import HumanMessage
#from langchain.agents.agent_types import AgentType
#from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
#from langchain_openai import ChatOpenAI
import pandas as pd
#from langchain_openai import OpenAI
from llama_index.llms import openai


import google.generativeai  as genai

import typing_extensions

from openai import OpenAI
import os


def create_session_state():
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 1.0
    
# Response Schema
class Command(typing_extensions.TypedDict):
    command: str
# These are the API calls that this chatbot can use
def getFileMetadata(file_name:str)->str:
    '''
    Returns all the metadata about a file, its size, name, etc. given a file name
    '''
    print ("Calling get file metadata")
    # The URL for our API calls
    url = 'https://proteomic.datacommons.cancer.gov/graphql'

    if file_name.find('.')>0:
        param = 'file_name:"' + file_name
    else:
        param = 'file_id:"' + file_name

      # query to get file metadata

    query = '''{
        fileMetadata('''
    query += param
    query += '''" acceptDUA: true) {
          file_name
          file_size
          md5sum
          file_location
          file_submitter_id
          fraction_number
          experiment_type
        }
      }'''
    response = requests.post(url, json={'query': query})
    print(response)
    if(response.ok):
        #If the response was OK then print the returned JSON
        jData = json.loads(response.content)
        jData = jData['data']['fileMetadata']
        # Add the query parameters
        for i in range(0, len(jData)):
            x = jData[i]
            x["file_name"] = file_name 
            jData[i] = x

        return json.dumps(jData, sort_keys=True)
    else:
      # If response code is not ok (200), print the resulting http error code with description
      response.raise_for_status()
      
# Get information about all the programs

def getProgramInformation()->str:
    '''
    Returns the information about a program like its start data and name
    '''

    query = '''
            { allPrograms {
                name
                start_date
            }
            }
            '''
    url = 'https://proteomic.datacommons.cancer.gov/graphql?query='+query
    response = requests.get(url)
    if(response.ok):
        #If the response was OK then print the returned JSON
        #If the response was OK then print the returned JSON
        jData = json.loads(response.content)
        return json.dumps(jData, sort_keys=True)
    else:
      # If response code is not ok (200), print the resulting http error code with description
      response.raise_for_status()
# Get case information for a given study
def getCaseInformation(case_id:str)->str:
    '''
    Get all the information for given case. A case id is really a case submitted ID.
    '''
    print("Calling getCaseInformation")
    query = ' { case (case_submitter_id:"' + case_id + '"' + ') {'
    query += ' case_submitter_id project_submitter_id days_to_lost_to_followup '
    query += ' disease_type index_date lost_to_followup primary_site consent_type days_to_consent '
    query += ' demographics{ demographic_id ethnicity gender demographic_submitter_id race cause_of_death days_to_birth '
    query += ' days_to_death vital_status year_of_birth year_of_death age_at_index premature_at_birth weeks_gestation_at_birth age_is_obfuscated '
    query += ' cause_of_death_source occupation_duration_years country_of_residence_at_enrollment } '
    query += ' samples { sample_id sample_submitter_id sample_type sample_type_id gdc_sample_id gdc_project_id biospecimen_anatomic_site composition current_weight '
    query += ' days_to_collection days_to_sample_procurement diagnosis_pathologically_confirmed freezing_method initial_weight intermediate_dimension longest_dimension method_of_sample_procurement '
    query += ' pathology_report_uuid preservation_method sample_type_id shortest_dimension time_between_clamping_and_freezing '
    query += ' time_between_excision_and_freezing tissue_type tumor_code tumor_code_id tumor_descriptor '
    query += ' biospecimen_laterality catalog_reference distance_normal_to_tumor distributor_reference growth_rate passage_count sample_ordinal tissue_collection_type '
    query += ' diagnoses{ diagnosis_id diagnosis_submitter_id annotation} aliquots { aliquot_id aliquot_submitter_id analyte_type  } } '
    query += ' diagnoses{ diagnosis_id tissue_or_organ_of_origin age_at_diagnosis primary_diagnosis tumor_grade '
    query += ' tumor_stage diagnosis_submitter_id classification_of_tumor days_to_last_follow_up '
    query += ' days_to_last_known_disease_status days_to_recurrence last_known_disease_status morphology '
    query += ' progression_or_recurrence site_of_resection_or_biopsy prior_malignancy ajcc_clinical_m ajcc_clinical_n ' 
    query += ' ajcc_clinical_stage ajcc_clinical_t ajcc_pathologic_m ajcc_pathologic_n ajcc_pathologic_stage '
    query += ' ajcc_pathologic_t ann_arbor_b_symptoms ann_arbor_clinical_stage ann_arbor_extranodal_involvement '
    query += ' ann_arbor_pathologic_stage best_overall_response burkitt_lymphoma_clinical_variant circumferential_resection_margin colon_polyps_history '
    query += ' days_to_best_overall_response days_to_diagnosis days_to_hiv_diagnosis days_to_new_event figo_stage hiv_positive hpv_positive_type hpv_status '
    query += ' iss_stage laterality ldh_level_at_diagnosis ldh_normal_range_upper lymph_nodes_positive lymphatic_invasion_present method_of_diagnosis new_event_anatomic_site new_event_type overall_survival perineural_invasion_present prior_treatment progression_free_survival progression_free_survival_event residual_disease vascular_invasion_present year_of_diagnosis icd_10_code synchronous_malignancy tumor_largest_dimension_diameter anaplasia_present anaplasia_present_type child_pugh_classification cog_liver_stage cog_neuroblastoma_risk_group cog_renal_stage cog_rhabdomyosarcoma_risk_group enneking_msts_grade enneking_msts_metastasis enneking_msts_stage enneking_msts_tumor_site esophageal_columnar_dysplasia_degree esophageal_columnar_metaplasia_present first_symptom_prior_to_diagnosis gastric_esophageal_junction_involvement goblet_cells_columnar_mucosa_present gross_tumor_weight inpc_grade inpc_histologic_group inrg_stage inss_stage irs_group irs_stage ishak_fibrosis_score lymph_nodes_tested medulloblastoma_molecular_classification metastasis_at_diagnosis metastasis_at_diagnosis_site mitosis_karyorrhexis_index peripancreatic_lymph_nodes_positive peripancreatic_lymph_nodes_tested supratentorial_localization tumor_confined_to_organ_of_origin tumor_focality tumor_regression_grade vascular_invasion_type wilms_tumor_histologic_subtype breslow_thickness gleason_grade_group igcccg_stage international_prognostic_index largest_extrapelvic_peritoneal_focus masaoka_stage non_nodal_regional_disease non_nodal_tumor_deposits ovarian_specimen_status ovarian_surface_involvement percent_tumor_invasion peritoneal_fluid_cytological_status primary_gleason_grade secondary_gleason_grade weiss_assessment_score adrenal_hormone ann_arbor_b_symptoms_described diagnosis_is_primary_disease eln_risk_classification figo_staging_edition_year gleason_grade_tertiary gleason_patterns_percent margin_distance margins_involved_site pregnant_at_diagnosis satellite_nodule_present sites_of_involvement tumor_depth who_cns_grade who_nte_grade samples { sample_id sample_submitter_id annotation}} exposures { exposure_id exposure_submitter_id alcohol_days_per_week alcohol_drinks_per_day alcohol_history alcohol_intensity asbestos_exposure cigarettes_per_day coal_dust_exposure environmental_tobacco_smoke_exposure pack_years_smoked radon_exposure respirable_crystalline_silica_exposure smoking_frequency time_between_waking_and_first_smoke tobacco_smoking_onset_year tobacco_smoking_quit_year tobacco_smoking_status type_of_smoke_exposure type_of_tobacco_used years_smoked age_at_onset, alcohol_type, exposure_duration, exposure_duration_years, exposure_type, marijuana_use_per_week, parent_with_radiation_exposure, secondhand_smoke_as_child, smokeless_tobacco_quit_age, tobacco_use_per_day} follow_ups {follow_up_id, follow_up_submitter_id, adverse_event, barretts_esophagus_goblet_cells_present, bmi, cause_of_response, comorbidity, comorbidity_method_of_diagnosis, days_to_adverse_event, days_to_comorbidity, days_to_follow_up, days_to_progression, days_to_progression_free, days_to_recurrence, diabetes_treatment_type, disease_response, dlco_ref_predictive_percent, ecog_performance_status, fev1_ref_post_bronch_percent, fev1_ref_pre_bronch_percent, fev1_fvc_pre_bronch_percent, fev1_fvc_post_bronch_percent, height, hepatitis_sustained_virological_response, hpv_positive_type, karnofsky_performance_status, menopause_status, pancreatitis_onset_year, progression_or_recurrence, progression_or_recurrence_anatomic_site, progression_or_recurrence_type, reflux_treatment_type, risk_factor, risk_factor_treatment, viral_hepatitis_serologies, weight, adverse_event_grade, aids_risk_factors, body_surface_area, cd4_count, cdc_hiv_risk_factors, days_to_imaging, evidence_of_recurrence_type, eye_color, haart_treatment_indicator, history_of_tumor, history_of_tumor_type, hiv_viral_load, hormonal_contraceptive_type, hormonal_contraceptive_use, hormone_replacement_therapy_type, hysterectomy_margins_involved, hysterectomy_type, imaging_result, imaging_type, immunosuppressive_treatment_type, nadir_cd4_count, pregnancy_outcome, procedures_performed, recist_targeted_regions_number, recist_targeted_regions_sum, scan_tracer_used, undescended_testis_corrected, undescended_testis_corrected_age, undescended_testis_corrected_laterality, undescended_testis_corrected_method, undescended_testis_history, undescended_testis_history_laterality} treatments {treatment_id, treatment_submitter_id, days_to_treatment_end, days_to_treatment_start, initial_disease_status, regimen_or_line_of_therapy, therapeutic_agents, treatment_anatomic_site, treatment_effect, treatment_intent_type, treatment_or_therapy, treatment_outcome, treatment_type, chemo_concurrent_to_radiation, number_of_cycles, reason_treatment_ended, route_of_administration, treatment_arm, '
    query += ' treatment_dose, treatment_dose_units, treatment_effect_indicator, treatment_frequency}}}'
    url = 'https://proteomic.datacommons.cancer.gov/graphql?query='+query
    response = requests.get(url)
    if(response.ok):
        #If the response was OK then print the returned JSON
        jData = json.loads(response.content)
        #return jData
        jData = jData['data']['case']
    return json.dumps(jData, sort_keys=True)   

# Get cases for a given study
def getStudyInformation(study_id:str)->str:
    '''
    Get case and other information like diagnoses, race, gender, ethincity, disease type for a give PDC study
    '''
    print("Calling getCaseForStudy")
    query = ' { paginatedCaseDemographicsPerStudy(pdc_study_id:"' + study_id + '"' + 'offset: 0 limit:1000) {'
    query += '''
        caseDemographicsPerStudy { case_id case_submitter_id disease_type primary_site 
            demographics { 
                demographic_id 
                ethnicity 
                gender 
                demographic_submitter_id 
                race 
                cause_of_death 
                days_to_birth 
                days_to_death 
                vital_status 
                year_of_birth 
                year_of_death 
                age_at_index 
                premature_at_birth 
                cause_of_death_source 
                occupation_duration_years 
                country_of_residence_at_enrollment
                } } 
                pagination { count sort from page total pages size } }}
            '''
    url = 'https://proteomic.datacommons.cancer.gov/graphql?query='+query
    response = requests.get(url)
    if(response.ok):
        #If the response was OK then print the returned JSON
        jData = json.loads(response.content)
        #return jData
        jData = jData['data']['paginatedCaseDemographicsPerStudy']['caseDemographicsPerStudy']
        # Add the query parameters
        for i in range(0, len(jData)):
            x = jData[i]
            x["pdc_study_id"] = study_id 
            jData[i] = x

        return json.dumps(jData, sort_keys=True)

    else:
      # If response code is not ok (200), print the resulting http error code with description
      response.raise_for_status()


file_tool = FunctionTool.from_defaults(fn=getFileMetadata, name="getFileMetadata")
program_tool = FunctionTool.from_defaults(fn=getProgramInformation, name="getProgramInformation")
study_tool = FunctionTool.from_defaults(fn=getStudyInformation, name="getStudyInformation")
case_tool = FunctionTool.from_defaults(fn=getCaseInformation, name="getCaseInformation")
tools = [file_tool, program_tool, study_tool, case_tool]


def processResponse(responses):
    all_responses = []
    for response in responses:
        for part in response.parts:
            if part.text:
                all_responses.append(part.text)
    return all_responses




def plotUsingOpenAI(user_query):
    response = llm.predict_and_call(
        tools,        
        user_query
        #verbose=True
    )

    input_data = json.loads(response.response)
    df = pd.DataFrame(input_data)
    if len(df.index)<1:
        print("No data found")
        return -1
    head = str(df.head().to_dict())
    desc = str(df.describe().to_dict())
    cols = str(df.columns.to_list())
    dtype = str(df.dtypes.to_dict())
    final_query = f"The dataframe name is 'df'. df has the columns {cols} and their datatypes are {dtype}. df is in the following format: {desc}. The head of df is: {head}. You cannot use df.info() or any command that cannot be printed. Do not use plt.figure() or print(). Write a python command for this query on the dataframe df: {user_query}"
    
    task = "Generate pandas code."
    task = task + " The script should only include code, no comments."

    # Call OpenAI API to generate code based on the prompt
    response = client.chat.completions.create(
        messages=[{"role":"system","content":task},{"role":"user","content":final_query}],
        model="gpt-4o-mini",
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.5)

    # Extract the generated code
    generated_code = response.choices[0].message.content
    #print(generated_code)
    try:
        lcls = locals()
        final_code = generated_code.replace("`", "")
        exec_code = final_code.replace("python", "")

        hack_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
        hack_code +="fig, ax = plt.subplots()\n" 
        hack_code += exec_code
        print(hack_code.strip())
        exec(f"{hack_code}", globals(), lcls)
        fig = lcls["fig"]
        plot_area = st.empty()
        plot_area.pyplot(fig)  
        return ""
    except Exception as e:
        print({"role": "assistant", "content": "Error"})
        print(e)
        return (["Failed to generate an answer."])
    #return generated_code


# This function is used to plot data using GeminiFlash
def plotUsingGemini(user_query):
    response = llm.predict_and_call(
        tools,        
        user_query
        #verbose=True
    )
    input_data = json.loads(response.response)
    df = pd.DataFrame(input_data)
    if len(df.index)<1:
        print("No data found")
        return -1
    head = str(df.head().to_dict())
    desc = str(df.describe().to_dict())
    cols = str(df.columns.to_list())
    dtype = str(df.dtypes.to_dict())
    final_query = f"The dataframe name is 'df'. df has the columns {cols} and their datatypes are {dtype}. df is in the following format: {desc}. The head of df is: {head}. You cannot use df.info() or any command that cannot be printed. Do not use plt.figure() in the generated code. Do not use print() in the generated code. Write a python command for this query on the dataframe df: {user_query}"
    responses = model_response.generate_content(
                    final_query,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=Command,
                        temperature=0.7
                    )
                )
    all_responses = processResponse(responses)
    #print("Responses:")
    #print(all_responses)
    command = json.loads(all_responses[0])['command']
    #command = str.replace(command, 'python', '')
    print("Command is:")
    print(command)

    try:
        lcls = locals()
#        command_prefix = "python\n"
        command_prefix = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
        command_prefix += "fig, ax = plt.subplots()\n"
        command = command_prefix + command
#        command += "\nfig"
        print("Command before exec:")
        print(command)

        exec(f"{command}", globals(), lcls)
        fig = lcls["fig"]
        plot_area = st.empty()
        plot_area.pyplot(fig)
        return ""

    except Exception as e:
        print({"role": "assistant", "content": "Error"})
        print(e)
        return (["Failed to generate an answer."])
# First create the dataframe which will be used to perform the analysis
# The LLM will select the appropriate tool based on the user question
def queryPDC(user_query):
    response = llm.predict_and_call(
        tools,        
        user_query
        #verbose=True
    )
    input_data = json.loads(response.response)
    df = pd.DataFrame(input_data)
    if len(df.index)<1:
        print("No data found")
        return -1
    head = str(df.head().to_dict())
    desc = str(df.describe().to_dict())
    cols = str(df.columns.to_list())
    dtype = str(df.dtypes.to_dict())
    #print(cols)
    final_query = f"The dataframe name is 'df'. df has the columns {cols} and their datatypes are {dtype}. df is in the following format: {desc}. The head of df is: {head}. You cannot use df.info() or any command that cannot be printed. Write a pandas command for this query on the dataframe df: {user_query}"
    responses = model_pandas.generate_content(
                    final_query,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=Command,
                        temperature=0.7
                    )
                )
    all_responses = processResponse(responses)
    command = json.loads(all_responses[0])['command']
    print(command)
    try:
        lcls = locals()
        exec(f"data = {command}", globals(), lcls)
        data = lcls["data"]

        #print(data)
        natural_response = f"The user query is {final_query}. The output of the command is {str(data)}. If the data is 'None', you can say 'Please ask a query to get started'. Do not mention the command used. Generate a response in natural language for the output."
        bot_response = model_response.generate_content(
            natural_response,
            generation_config=genai.GenerationConfig(temperature=0.7)
        )
        all_responses = processResponse(bot_response)
        return all_responses

    except Exception as e:
        print({"role": "assistant", "content": "Error"})
        print(e)
        return (["Failed to generate an answer."])

#strQuery = "list all the different disease types for study PDC000448"
#user_query = "list all the different disease types for study PDC000448"
#queryPDC(strQuery)

#---- End Setup --
#getData("list all the case submitter ids for non reference cases with ajcc pathologic stage of II or higher for study PDC0005")
#---- UI Code ---

#create_session_state()
def reset_session():
    st.session_state['temperature'] = 0.0
    st.session_state['token_limit'] = 256
    st.session_state['top_k'] = 40
    st.session_state['top_p'] = 0.8
    st.session_state['debug_mode'] = False
    st.session_state['return_references'] = False
    st.session_state['prompt'] = []
    st.session_state['response'] = []
    st.session_state['refs'] = []
    st.session_state['model_used'] = 'gpt-turbo'
    st.session_state['messages'] = [{"role": "assistant", "content": "Ask any question related to PDC studies"}]

st.set_page_config(
    page_title="PDC API Exploration using LLMs",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This app uses Google's Gemini-Flash and gpt-turbo-3.5 to answer questions about PDC"
    }
)

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: red;'>Setting 🎈</h2>", unsafe_allow_html=True)
    openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    gemini_api_key = st.text_input("Google Gemini API Key", key="gemini_api_key", type="password")
    
    #define the temeperature for the model
    temperature_value = st.slider('Temperature :', 0.0, 1.0, 0.2)
    st.session_state['temperature'] = temperature_value

    #define the tokens for the model
    token_limit_value = st.slider('Token limit :', 1, 20000, 256)
    st.session_state['token_limit'] = token_limit_value
    
    # define the model to use
    model_used = st.radio('Use Model for Plotting:', options=['gpt-turbo', 'gemini-flash'], captions=["GPT-Turbo-3.5", "Gemini-Flash-1.5"], index=1)
    st.session_state['model_used'] = model_used
    if st.button("Reset Session"):
        reset_session()

st.title('💬 A PDC Chatbot using PDC APIs')
st.header(":blue[Using ChatGPT-4 and Gemini-Flash]")
st.subheader('I can only answer questions that have a PDC study ID specified or a File name or ID or a case submitter id. Study IDs are like PDC000xxx')
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask any question related to PDC studies"}]
    firstQuery = True
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if prompt := st.chat_input():
    if not st.session_state['openai_api_key']:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if not st.session_state['gemini_api_key']:
        st.info("Please add your Google Gemini API key to continue.")
        st.stop()
    # This is the default model from openai - gpt4o
    client = OpenAI(api_key = openai_api_key)

    # This is a specific model from llama_index library
    # TO-DO: Use the same models
    llm = openai.OpenAI(api_key = openai_api_key, model="gpt-3.5-turbo")
    
    ##CHANGE THIS
    #gemini_key='AIzaSyCvDS90ATrK30_eFQSt8CpSR43u3puRjKw'
    genai.configure(api_key=gemini_api_key)

    model_pandas = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction="You are an expert python developer who works with pandas. You make sure to generate simple pandas 'command' for the user queries in JSON format. No need to add 'print' function. Analyse the datatypes of the columns before generating the command. If unfeasible, return 'None'. ")
    model_response = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction="Your task is to comprehend. You must analyse the user query and response data to generate a response data in natural language.")
    model_plotly = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction="You are an expert python developer who works with matplotlib and pandas. You make sure to generate simple python 'code' for the user queries in JSON format. No need to add 'print' function. Analyse the datatypes of the columns before generating the code. If unfeasible, return 'None'. ")

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # If they just want to print something, use the vanilla queryPDC
    if prompt.lower().find("visualize")==-1 and prompt.lower().find("draw")==-1 and prompt.lower().find("plot")==-1: 
        response=queryPDC(prompt)
        if response == -1:
            msg = "No data found"
        else:
            msg = str(response[0])
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg) 
    else:
        if st.session_state['model_used'] == 'gpt-turbo':
            print("Using OpenAI")
            response = plotUsingOpenAI(prompt)
        else:
            print("Using Gemini model")
            response = plotUsingGemini(prompt)

        if response==-1:
            msg = "No data found"   
        else:
            msg = str(response)
        st.chat_message("assistant").write(msg)







