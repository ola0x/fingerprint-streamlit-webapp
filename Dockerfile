FROM python:3.8-slim-buster

RUN apt-get update -y --no-install-recommends

# gcc compiler and opencv prerequisites
RUN apt-get -y --no-install-recommends install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build && \
  rm -rf /var/lib/apt/lists/*

# Detectron2 prerequisites
RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html

# Development packages
RUN pip install flask flask-cors requests opencv-python streamlit scipy

# Added after testing the flask code
WORKDIR /app
COPY app.py /app/app.py

COPY crop_finger.py /app/crop_finger.py

COPY FingerprintImageEnhancer.py /app/FingerprintImageEnhancer.py

COPY inference.py /app/inference.py

COPY model_final.pth /app/model_final.pth

EXPOSE 8501

RUN ls -la /app/

ENTRYPOINT ["streamlit", "run"]

CMD ["/app/app.py"]