# Select Python Docker Image (v3.8)
FROM python:3.8

# Perform necessary updates to set up the Docker container
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc ffmpeg libsm6 libxext6 && \
    apt clean && rm -rf /var/lib/apt/lists/*


# Copy necessary directories to the Docker container
COPY ./code /code
COPY ./secret_weapon /secret_weapon
COPY ./results/models/visum2022.pt results/models/visum2022.pt


# Install Python libraries
RUN pip3 install -U albumentations torchinfo tqdm pycocotools argparse
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Rest of libs for the secret weapon
RUN pip3 install opencv-python cython pillow
RUN cd /secret_weapon/utils/box/ext/rbbox_overlap_cpu && rm -rf rbbox_overlap_cpu *.so *.cpp && python3 setup.py build_ext --inplace && cp rbbox_overlap_cpu/* .
#RUN cd /code &&  ./scripts/masks_to_coord.py data/json/challenge/test_challenge.json data/masks data/json/challenge/test_box_labels.npy --test
RUN ln -s /code/data /secret_weapon/data
WORKDIR /secret_weapon

# Run command to test the participants' model and generate the score
#CMD ["python3", "/code/model_test.py"]
CMD ["python3", "/secret_weapon/run/visum/test.py", "resnet50"]
