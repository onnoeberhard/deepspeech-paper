# Working directory
mkdir deepspeech
cd deepspeech

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
rm ./Miniconda3-latest-Linux-x86_64.sh

# Create conda envionment and install pip dependencies
conda create -n deepspeech python=3.6
conda activate deepspeech
pip install audiomate num2words progressbar2 'tensorflow-gpu<2.0'

# Clone and install DeepSpeech
git clone https://github.com/mozilla/DeepSpeech.git
cd DeepSpeech
git reset --hard v0.7.4
pip install -e .
cd ..

# Install conda dependencies (need to be installed after all pip-dependencies including DeepSpeech)
conda install -c conda-forge sox librosa    # These two lines are trouble makers: remove if not absolutely necessary
conda install -c anaconda cmake boost       # *
conda install cudatoolkit=10.0 cupti=10.0 cudnn=7.6

# Clone deepspeech-german
git clone https://github.com/AASHISHAG/deepspeech-german.git

# Download and extract corpus
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz
gzip -d German_sentences_8mil_filtered_maryfied.txt.gz
mv German_sentences_8mil_filtered_maryfied.txt corpus.txt

# Prepare text data
python deepspeech-german/pre-processing/prepare_vocab.py corpus.txt vocab.txt
rm corpus.txt

# Download and install KenLM
wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
mkdir -p kenlm/build
cd kenlm/build
cmake ..
make -j `nproc`
cd -

# Build language model and scorer
python DeepSpeech/data/lm/generate_lm.py --input_txt vocab.txt \
    --output_dir . --top_k 500000 --kenlm_bins kenlm/build/bin \
    --arpa_order 5 --max_arpa_memory "50%" --arpa_prune "0|0|1" \    # Suggested 85% memory results in SIGKILL on server
    --binary_a_bits 255 --binary_q_bitechos 8 --binary_type trie
python DeepSpeech/data/lm/generate_package.py --alphabet deepspeech-german/data/alphabet.txt \
    --lm lm.binary --vocab vocab-500000.txt --package kenlm.scorer \
    --default_alpha 0.931289039105002 --default_beta 1.1834137581510284

# Download and extract speech data (Mozilla Common Voice 2.0)
wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-2/de.tar.gz
mkdir common_voice
tar -xvf de.tar.gz -C common_voice
rm de.tar.gz

# Prepare audio data
python DeepSpeech/bin/import_cv2.py --filter_alphabet deepspeech-german/data/alphabet.txt common_voice

# Remove files from previous trainings
rm -rf ~/.local/share/deepspeech/summaries
rm -rf ~/.local/share/deepspeech/checkpoints

# Train model
nohup python DeepSpeech/DeepSpeech.py --train_files common_voice/clips/train.csv --dev_files common_voice/clips/dev.csv --test_files common_voice/clips/test.csv --alphabet_config_path deepspeech-german/data/alphabet.txt --scorer kenlm.scorer --test_batch_size 36 --train_batch_size 24 --dev_batch_size 36 --epochs 30 --learning_rate 0.0005 --dropout_rate 0.40 --export_dir models </dev/null >training.log 2>&1 &

# View output of training process
tail -f training.log

# --- SECOND PHASE --- (still inside deepspeech working dir with deepspeech conda env activated)
# Set up working directory
mkdir ../simple-transfer
mv -t ../simple-transfer common_voice kenlm.scorer DeepSpeech
cp deepspeech-german/data/alphabet.txt ../simple-transfer
cd ../simple-transfer

# Download pretrained model checkpoint
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.7.4/deepspeech-0.7.4-checkpoint.tar.gz
tar -xvf deepspeech-0.7.4-checkpoint.tar.gz
mv deepspeech-0.7.4-checkpoint pretrained
rm deepspeech-0.7.4-checkpoint.tar.gz

# Remove files from previous trainings
rm -rf savecheckpoint

# Commence training
nohup DeepSpeech/DeepSpeech.py \
    --drop_source_layers 1 \
    --alphabet_config_path alphabet.txt \
    --save_checkpoint_dir savecheckpoint \
    --load_checkpoint_dir pretrained \
    --train_files common_voice/clips/train.csv \
    --dev_files common_voice/clips/dev.csv \
    --test_files common_voice/clips/test.csv \
    --test_batch_size 36 \
    --train_batch_size 24 \
    --dev_batch_size 36 \
    --epochs 30 \
    --learning_rate 0.0005 \
    --dropout_rate 0.40 \
    --scorer kenlm.scorer \
    --export_dir models \
    --train_cudnn \
    </dev/null >training.log 2>&1 &

# View output of training process
tail -f training.log

# Testing
nohup DeepSpeech/DeepSpeech.py \
    --alphabet_config_path alphabet.txt \
    --save_checkpoint_dir savecheckpoint_test \
    --load_checkpoint_dir savecheckpoint \
    --test_files common_voice/clips/test.csv \
    --test_batch_size 36 \
    --scorer kenlm.scorer \
    </dev/null >testing.log 2>&1 &

# View output of testing process
tail -f training.log

# --- THRID PHASE --- (still inside simple-transfer working dir with deepspeech conda env activated)
# Set up working directory
mkdir ../finetune
mv -t ../finetune common_voice kenlm.scorer alphabet.txt pretrained
cd ../finetune

# Create conda envionment
conda create -n ds-finetune python=3.6
conda activate ds-finetune

# Clone and install deepspeech-transfer and install pip dependencies
git clone https://github.com/onnoeberhard/deepspeech-transfer
cd deepspeech-transfer
git checkout transfer
pip install -e .
pip install audiomate num2words progressbar2 'tensorflow-gpu<2.0'
cd ..

# Install conda dependencies (need to be installed after all pip-dependencies including DeepSpeech)
conda install cudatoolkit=10.0 cupti=10.0 cudnn=7.6

# Remove files from previous trainings
rm -rf ~/.local/share/deepspeech/summaries
rm -rf ~/.local/share/deepspeech/checkpoints

# Commence training
nohup deepspeech-transfer/DeepSpeech.py \
    --drop_source_layers 1 \
    --alphabet_config_path alphabet.txt \
    --save_checkpoint_dir savecheckpoint \
    --load_checkpoint_dir pretrained \
    --train_files common_voice/clips/train.csv \
    --dev_files common_voice/clips/dev.csv \
    --test_files common_voice/clips/test.csv \
    --test_batch_size 36 \
    --train_batch_size 24 \
    --dev_batch_size 36 \
    --epochs 30 \
    --learning_rate 0.0005 \
    --dropout_rate 0.40 \
    --scorer kenlm.scorer \
    --export_dir models \
    --train_cudnn \
    </dev/null >training.log 2>&1 &

# View output of training process
tail -f training.log

# Testing
nohup deepspeech-transfer/DeepSpeech.py \
    --alphabet_config_path alphabet.txt \
    --save_checkpoint_dir savecheckpoint_test \
    --load_checkpoint_dir savecheckpoint \
    --test_files common_voice/clips/test.csv \
    --test_batch_size 36 \
    --scorer kenlm.scorer \
    </dev/null >testing.log 2>&1 &

# View output of testing process
tail -f training.log

# --- FOURTH PHASE --- (still inside finetune working dir with ds-finetune conda env activated)
# Set up working directory
mkdir ../finetune-1layer
mv -t ../finetune-1layer common_voice kenlm.scorer alphabet.txt pretrained
cd ../finetune

# Clone and install deepspeech-transfer and install pip dependencies
git clone https://github.com/onnoeberhard/deepspeech-transfer
cd deepspeech-transfer
git checkout transfer-1
pip install -e .
cd ..

# Remove files from previous trainings
rm -rf ~/.local/share/deepspeech/summaries
rm -rf ~/.local/share/deepspeech/checkpoints

# Commence training
nohup deepspeech-transfer/DeepSpeech.py \
    --drop_source_layers 1 \
    --alphabet_config_path alphabet.txt \
    --save_checkpoint_dir savecheckpoint \
    --load_checkpoint_dir pretrained \
    --train_files common_voice/clips/train.csv \
    --dev_files common_voice/clips/dev.csv \
    --test_files common_voice/clips/test.csv \
    --test_batch_size 36 \
    --train_batch_size 24 \
    --dev_batch_size 36 \
    --epochs 30 \
    --learning_rate 0.0005 \
    --dropout_rate 0.40 \
    --scorer kenlm.scorer \
    --export_dir models \
    --train_cudnn \
    </dev/null >training.log 2>&1 &

# View output of training process
tail -f training.log

# Testing
nohup deepspeech-transfer/DeepSpeech.py \
    --alphabet_config_path alphabet.txt \
    --save_checkpoint_dir savecheckpoint_test \
    --load_checkpoint_dir savecheckpoint \
    --test_files common_voice/clips/test.csv \
    --test_batch_size 36 \
    --scorer kenlm.scorer \
    </dev/null >testing.log 2>&1 &

# View output of testing process
tail -f testing.log

# --- FIFTH PHASE --- (still inside finetune-1layer working dir with ds-finetune conda env activated)
# Set up working directory
mkdir ../finetune-2layer
mv -t ../finetune-2layer common_voice kenlm.scorer alphabet.txt pretrained
cd ../finetune-2layer

# Clone and install deepspeech-transfer and install pip dependencies
git clone https://github.com/onnoeberhard/deepspeech-transfer
cd deepspeech-transfer
git checkout transfer-2
pip install -e .
cd ..

# Remove files from previous trainings
rm -rf ~/.local/share/deepspeech/summaries
rm -rf ~/.local/share/deepspeech/checkpoints

# Commence training
nohup deepspeech-transfer/DeepSpeech.py \
    --drop_source_layers 1 \
    --alphabet_config_path alphabet.txt \
    --save_checkpoint_dir savecheckpoint \
    --load_checkpoint_dir pretrained \
    --train_files common_voice/clips/train.csv \
    --dev_files common_voice/clips/dev.csv \
    --test_files common_voice/clips/test.csv \
    --test_batch_size 36 \
    --train_batch_size 24 \
    --dev_batch_size 36 \
    --epochs 30 \
    --learning_rate 0.0005 \
    --dropout_rate 0.40 \
    --scorer kenlm.scorer \
    --export_dir models \
    --train_cudnn \
    </dev/null >training.log 2>&1 &

# View output of training process
tail -f training.log

# Testing
nohup deepspeech-transfer/DeepSpeech.py \
    --alphabet_config_path alphabet.txt \
    --save_checkpoint_dir savecheckpoint_test \
    --load_checkpoint_dir savecheckpoint \
    --test_files common_voice/clips/test.csv \
    --test_batch_size 36 \
    --scorer kenlm.scorer \
    </dev/null >testing.log 2>&1 &

# View output of testing process
tail -f testing.log

# --- SIXTH PHASE --- (still inside finetune-2layer working dir with ds-finetune conda env activated)
# Set up working directory
mkdir ../finetune-4layer
mv -t ../finetune-4layer common_voice kenlm.scorer alphabet.txt pretrained
cd ../finetune-4layer

# Clone and install deepspeech-transfer and install pip dependencies
git clone https://github.com/onnoeberhard/deepspeech-transfer
cd deepspeech-transfer
git checkout transfer-4
pip install -e .
cd ..

# Remove files from previous trainings
rm -rf ~/.local/share/deepspeech/summaries
rm -rf ~/.local/share/deepspeech/checkpoints

# Commence training
nohup deepspeech-transfer/DeepSpeech.py \
    --drop_source_layers 1 \
    --alphabet_config_path alphabet.txt \
    --save_checkpoint_dir savecheckpoint \
    --load_checkpoint_dir pretrained \
    --train_files common_voice/clips/train.csv \
    --dev_files common_voice/clips/dev.csv \
    --test_files common_voice/clips/test.csv \
    --test_batch_size 36 \
    --train_batch_size 24 \
    --dev_batch_size 36 \
    --epochs 30 \
    --learning_rate 0.0005 \
    --dropout_rate 0.40 \
    --scorer kenlm.scorer \
    --export_dir models \
    --train_cudnn \
    </dev/null >training.log 2>&1 &

# View output of training process
tail -f training.log

# Testing
nohup deepspeech-transfer/DeepSpeech.py \
    --alphabet_config_path alphabet.txt \
    --save_checkpoint_dir savecheckpoint_test \
    --load_checkpoint_dir savecheckpoint \
    --test_files common_voice/clips/test.csv \
    --test_batch_size 36 \
    --scorer kenlm.scorer \
    </dev/null >testing.log 2>&1 &

# View output of testing process
tail -f testing.log