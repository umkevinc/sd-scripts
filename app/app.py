import os
import subprocess
import streamlit as st
import uuid
# import extra_streamlit_components as stx
from natsort import os_sorted
from streamlit_js_eval import streamlit_js_eval

from glob import glob

# Configuration
st.set_page_config(layout="wide")

TARGET_DATA_ROOT = '/home/gazai/opt/DATA/ft_inputs'
DATASET_CONFIG_ROOT = '/home/gazai/opt/DATA/dataset_configs'

UUID = uuid.uuid1()


# Heulper functions
def display_folder(target_path):
    elms = glob(os.path.join(target_path, '*'))
    st.code('\n'.join(os_sorted(elms)))
    options = {os.path.split(elm)[1]: elm for elm in elms}
    return options


def list_target_folder(target_path):
    from streamlit.components.v1 import html
    elms = glob(os.path.join(target_path, '**'), recursive=True)
    elms = [elm.replace(target_path + '/', '') for elm in elms if elm.replace(target_path + '/', '')]
    elms = os_sorted(elms)
    
    files_str = '\n'.join(elms)
    # text = (
    #     f"""
    #     <code style="background-color:rgba(255,255,255,0.1)">
    #     {files_str}</code>
    #     """)
    # html(text, height=300, scrolling=True)
    st.code('\n'.join(elms), line_numbers=True)


def mkdir(target_dir):
    import pathlib
    if target_dir:
        pathlib.Path(target_dir).mkdir(parents=False, exist_ok=False)


def upload_files(outpu_dir):
    # st.write(output_dir)
    uploaded_files = st.file_uploader("Upload files:", accept_multiple_files=True, key="file_upload")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()    
            # st.write("filename:", uploaded_file.name)
            output_path = os.path.join(output_dir, uploaded_file.name)
            # st.write("output path:", output_path)
            with open(output_path, "wb") as binary_file:
                # Write bytes to file
                binary_file.write(bytes_data)
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

def upload_dataset_config_files(output_dir):
    # st.write(output_dir)
    dataset_config_files = st.file_uploader("Upload files:", accept_multiple_files=True, key="dataset_upload")
    if dataset_config_files:
        for uploaded_file in dataset_config_files:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()    
            # st.write("filename:", uploaded_file.name)
            output_path = os.path.join(output_dir, uploaded_file.name)            
            with open(output_path, "wb") as binary_file:
                # Write bytes to file
                binary_file.write(bytes_data)
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
        

# with st.sidebar:    
st.title('🎛 Gazai Finetune')

tab1, tab2 = st.tabs(["File Upload", "Fine-tune"])

####### First Tab #######
with tab1:
    col1, col2,col3 = st.columns(3)
    with col1:
        st.subheader("1. Create new project folder")
        st.text('Current projects:')
        target_folder_options = display_folder(TARGET_DATA_ROOT)
        target_project_name = st.text_input('Create a new project:', placeholder='e.g. umako_gilr_v1')
        if target_project_name:
            target_project_path = os.path.join(TARGET_DATA_ROOT, target_project_name).strip()
            st.write(target_project_path)

            if os.path.exists(target_project_path):
                st.warn(f'Error: project path exists! {target_project_path}')
            else:            
                mkdir(target_project_path)
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
        
    with col2: 
        st.subheader("2. Upload training data")
        st.caption("Please upload the training images and caption files to target folder.")
        selected_target_folder = st.selectbox(
                "Select a target folder:",
                [''] + os_sorted(list(target_folder_options.keys())))
        if selected_target_folder:
            output_dir = target_folder_options[selected_target_folder]
            upload_files(output_dir)
            list_target_folder(output_dir)
    with col3: 
        st.subheader("3. Upload dataset_config")
        st.caption(f"dataset_config root: {DATASET_CONFIG_ROOT}")        
        upload_dataset_config_files(DATASET_CONFIG_ROOT)
        list_target_folder(DATASET_CONFIG_ROOT)
        st.caption("Note. dataset_config format, please check: https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-ja.md")        
        
####### Second Tab ####### 
def get_basic_training(output_dir):
    pretrained_model_name_or_path = st.text_input(
        "pretrained_model_name_or_path:",
        value="/home/gazai/MyPrograms/a1111/stable-diffusion-webui/models/Stable-diffusion/AnythingV5Ink_ink.safetensors")
    dataset_config = st.text_input(
        "dataset_config:",    
        value="/home/gazai/opt/DATA/dataset_configs/plu.toml")
    model_output_name = st.text_input(
        "model_output_name:",
        value="plu-ft-v0-1ks")
    optimizer_type = st.text_input(
        "optimizer_type:",
        value="AdamW8bit")
    mixed_precision = st.text_input(
        "mixed_precision:",
        value="fp16")
    max_train_steps = int(st.number_input(
        "max_train_steps:",
        step=100, min_value=100, max_value=2000, value=1500))

    # Not configurable
    
    save_model_as = "safetensors"
    
    basic_training = f"""accelerate launch --num_cpu_threads_per_process 1 train_db.py \
    --pretrained_model_name_or_path={pretrained_model_name_or_path} \
    --dataset_config={dataset_config} \
    --output_dir={output_dir} \
    --output_name={model_output_name} \
    --save_model_as={save_model_as} \
    --prior_loss_weight=1.0 \
    --resolution=512 \
    --max_train_steps={max_train_steps} \
    --learning_rate=1e-6 \
    --optimizer_type={optimizer_type} \
    --xformers \
    --mixed_precision={mixed_precision} \
    --cache_latents \
    --gradient_checkpointing 
    """
    return basic_training

def run_and_display_training_stdout(*cmd_with_args, cwd='/home/gazai/workspace/sd-scripts'):
    result = subprocess.Popen(cmd_with_args, stdout=subprocess.PIPE, cwd=cwd)
    for line in iter(lambda: result.stdout.readline(), b""):
        st.caption(line.decode("utf-8"))  
        
with tab2:
    st.subheader("Basic Finetune")
    model_output_dir = "/home/gazai/opt/DATA/model_output"        
    basic_training = get_basic_training(model_output_dir)
    st.text("[Command]:")
    st.text(basic_training)
    start_button = st.button("Run Finetune")
    if start_button:
        run_and_display_training_stdout(*basic_training.split())
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
    
    st.subheader("Model Output Location")
    display_folder(model_output_dir)










