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
FINETUNE_MODEL_OUTPUT_DIR = "/home/gazai/opt/DATA/model_output"
MODEL_BASE_PATH = '/home/gazai/opt/stable-diffusion-webui/models/Stable-diffusion/'

UUID = uuid.uuid1()


# Heulper functions
def run_and_display_training_stdout(*cmd_with_args, cwd='/home/gazai/workspace/sd-scripts'):
    result = subprocess.Popen(cmd_with_args, stdout=subprocess.PIPE, cwd=cwd)
    for line in iter(lambda: result.stdout.readline(), b""):
        st.caption(line.decode("utf-8"))


def get_base_models():
    model_path_lst = glob(os.path.join(MODEL_BASE_PATH, '*.safetensors'))
    models = [os.path.split(elm)[1] for elm in model_path_lst]
    return models


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

tab1, tab2, tab3, tab4 = st.tabs(["File Upload", "Fine-tune", "SDXL-Lora-DB (Kevin)", "SDXL-Lora-DB (config-based)"])

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

####### Third Tab #######
def get_sdxl_lora_training():
    # Base model
    model_base_path = MODEL_BASE_PATH
    base_models = glob(f'{model_base_path}/*.safetensors')
    # model_names = [(os.path.split(path)[1]) for path in base_models]
    model_names = get_base_models()
    default_ix = model_names.index('bluePencilXL_v200.safetensors')
    selected_model = st.selectbox("base_model:", model_names, index=default_ix)
    if selected_model:
        pretrained_model_name_or_path = os.path.join(model_base_path, selected_model)
        st.text(pretrained_model_name_or_path)

    # Training data
    train_data_basedir = '/home/gazai/opt/DATA/ft_inputs'
    train_data_dir_options = ['<Please select>'] + [elm for elm in os.listdir(train_data_basedir) if not elm.startswith('reg_')]
    default_ix2 = train_data_dir_options.index('plu_train_sdxl')
    selected_data_dir = st.selectbox('SDXL training data:', train_data_dir_options, index=default_ix2)
    train_data_dir = os.path.join(train_data_basedir, selected_data_dir)
    st.text(train_data_dir)

    # Regulerization data dir
    reg_data_dir_options = [os.path.split(elm)[1] for elm in glob('/home/gazai/opt/DATA/ft_inputs/reg_*')]
    default_ix3 = reg_data_dir_options.index('reg_gen_girl_sdxl')
    select_reg_data_dir = st.selectbox('SDXL regularization data:', reg_data_dir_options, index=default_ix3)
    reg_data_dir = os.path.join(select_reg_data_dir, select_reg_data_dir)
    st.text(reg_data_dir)

    model_output_name = st.text_input(
        "SDXL model_output_name:",
        placeholder="eg. plu-sdxl-v0")

    # Not configurable
    save_model_as = "safetensors"
    learning_rate="0.0001" #Learning rate. Remember this is supposed to be a magnitude larger than a dreambooth equivalent. Worked well for me at this rate.
    text_encoder_lr="0.00005" #Learning rate for TEXT ENCODER. This is the value suggested in the ninja scrolls. Seems to work better for details.
    train_batch_size="2" #Amount of images to process at once. I have 8GB of VRAM so I left it at 1, it just worked. Raise if you got more VRAM.
    num_epochs="6" #Total number of epochs (amount of times the entire set is repeated)
    save_every_x_epochs="2" #Save checkpoints every X epochs.
    network_dim="160" #Higher for more resemblance to the training images and bigger file size. 96-192 for characters.
    scheduler="cosine_with_restarts"

    sdxl_training = f"""accelerate launch --num_cpu_threads_per_process 8 \
    sdxl_train_network.py  \
    --network_module=networks.lora \
    --pretrained_model_name_or_path={pretrained_model_name_or_path} \
    --train_data_dir={train_data_dir} \
    --reg_data_dir={reg_data_dir} \
    --output_dir={FINETUNE_MODEL_OUTPUT_DIR}/LORA \
    --output_name={model_output_name}_last_e{num_epochs}_n{network_dim} \
    --caption_extension=.txt \
    --shuffle_caption \
    --prior_loss_weight=1 \
    --network_alpha={network_dim}  \
    --resolution=1024 \
    --enable_bucket \
    --min_bucket_reso=768 \
    --max_bucket_reso=1024 \
    --train_batch_size={train_batch_size}  \
    --gradient_accumulation_steps=1 \
    --learning_rate={learning_rate}\
    --unet_lr={learning_rate} \
    --text_encoder_lr={text_encoder_lr} \
    --max_train_epochs={num_epochs} \
    --mixed_precision=fp16 \
    --save_precision=fp16 \
    --use_8bit_adam \
    --gradient_checkpointing \
    --xformers  \
    --save_every_n_epochs={save_every_x_epochs} \
    --save_model_as=safetensors \
    --clip_skip=2 \
    --seed=420  \
    --flip_aug \
    --color_aug \
    --face_crop_aug_range=2.0,4.0  \
    --network_dim={network_dim} \
    --max_token_length=150  \
    --lr_scheduler={scheduler} \
    --training_comment=LORA:{model_output_name}
    """

    return sdxl_training

with tab3:
    st.subheader("SDXL Lora Finetune")
    sdxl_lora_training = get_sdxl_lora_training()
    st.text("[Command]:")
    st.text(sdxl_lora_training)
    start_button = st.button("Run Finetune", key="sdxl4")
    if start_button:
        run_and_display_training_stdout(*sdxl_lora_training.split())
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

    st.subheader("Model Output Location")
    display_folder(f"{FINETUNE_MODEL_OUTPUT_DIR}/LORA")

####### Fourth Tab #######
def get_sdxl_lora_training_2():
    # Base model
    model_base_path = MODEL_BASE_PATH
    base_models = glob(f'{model_base_path}/*.safetensors')
    # model_names = [(os.path.split(path)[1]) for path in base_models]
    model_names = get_base_models()
    default_ix = model_names.index('bluePencilXL_v200.safetensors')
    selected_model = st.selectbox("base_model:", model_names, index=default_ix, key="sdxl_sb1")
    if selected_model:
        pretrained_model_name_or_path = os.path.join(model_base_path, selected_model)
        st.text(pretrained_model_name_or_path)

    # Dataset config
    dataset_config = st.text_input(
        "dataset_config:",
        value="/home/gazai/opt/DATA/dataset_configs/navyguy.toml")

    model_output_name = st.text_input(
        "SDXL model_output_name:",
        placeholder="eg. plu-sdxl-v1")

    # Not configurable
    save_model_as = "safetensors"
    learning_rate="0.0001" #Learning rate. Remember this is supposed to be a magnitude larger than a dreambooth equivalent. Worked well for me at this rate.
    text_encoder_lr="0.00005" #Learning rate for TEXT ENCODER. This is the value suggested in the ninja scrolls. Seems to work better for details.
    train_batch_size="2" #Amount of images to process at once. I have 8GB of VRAM so I left it at 1, it just worked. Raise if you got more VRAM.
    num_epochs="6" #Total number of epochs (amount of times the entire set is repeated)
    save_every_x_epochs="2" #Save checkpoints every X epochs.
    network_dim="160" #Higher for more resemblance to the training images and bigger file size. 96-192 for characters.
    scheduler="cosine_with_restarts"

    sdxl_training = f"""accelerate launch --num_cpu_threads_per_process 8 \
    sdxl_train_network.py  \
    --network_module=networks.lora \
    --pretrained_model_name_or_path={pretrained_model_name_or_path} \
    --dataset_config={dataset_config} \
    --output_dir={FINETUNE_MODEL_OUTPUT_DIR}/LORA \
    --output_name={model_output_name}_last_e{num_epochs}_n{network_dim} \
    --caption_extension=.txt \
    --shuffle_caption \
    --prior_loss_weight=1 \
    --network_alpha={network_dim}  \
    --resolution=1024 \
    --enable_bucket \
    --min_bucket_reso=768 \
    --max_bucket_reso=1024 \
    --train_batch_size={train_batch_size}  \
    --gradient_accumulation_steps=1 \
    --learning_rate={learning_rate}\
    --unet_lr={learning_rate} \
    --text_encoder_lr={text_encoder_lr} \
    --max_train_epochs={num_epochs} \
    --mixed_precision=fp16 \
    --save_precision=fp16 \
    --use_8bit_adam \
    --gradient_checkpointing \
    --xformers  \
    --save_every_n_epochs={save_every_x_epochs} \
    --save_model_as=safetensors \
    --clip_skip=2 \
    --seed=420  \
    --color_aug \
    --face_crop_aug_range=2.0,4.0  \
    --network_dim={network_dim} \
    --max_token_length=150  \
    --lr_scheduler={scheduler} \
    --training_comment=LORA:{model_output_name}
    """

    return sdxl_training

with tab4:
    st.subheader("SDXL Lora Finetune")
    sdxl_lora_training = get_sdxl_lora_training_2()
    st.text("[Command]:")
    st.text(sdxl_lora_training)
    start_button = st.button("Run Finetune", key="sdxl5")
    if start_button:
        run_and_display_training_stdout(*sdxl_lora_training.split())
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

    st.subheader("Model Output Location")
    display_folder(f"{FINETUNE_MODEL_OUTPUT_DIR}/LORA")
